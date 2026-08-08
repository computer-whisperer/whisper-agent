//! Driver-declared configuration knobs (migration step 10).
//!
//! A scripted driver optionally defines `describe()` beside
//! `on_event`/`present` — a pure, argument-free entry point returning the
//! driver's display metadata and a flat list of typed configuration knobs.
//! The new-thread form fetches this via `DescribeDriver`, renders one
//! control per knob, and submits the chosen values inside
//! `ThreadDriverConfig::Scripted { config }`. The server validates the
//! values against the declaration at creation (refusing on mismatch),
//! materializes declaration defaults, and freezes the result onto the
//! weave — the driver then receives it as the third argument of every
//! `on_event`/`present` call.
//!
//! The vocabulary is deliberately a typed knob list, not a general schema
//! language: every kind maps onto a client widget the form already has,
//! and `model` is a domain type (a `{backend, model}` pair riding the
//! existing backend/model pickers) that no structural schema could
//! express. Validation here is structural only; the server layers the
//! pod-ceiling backend check on `model` values where it has the catalog.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Kind tag of one declared knob. Determines the client widget and the
/// value shape accepted at creation.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KnobKind {
    /// Value: `{ "backend": <name>, "model": <id> }`, both non-empty
    /// strings. Rendered with the client's backend + model pickers; the
    /// server additionally checks the backend against the pod's
    /// `allow.backends` at creation.
    Model,
    /// Value: any string. `multiline` requests a textarea-style control
    /// (system prompts and the like).
    String,
    /// Value: `true` / `false`.
    Boolean,
    /// Value: a whole number, optionally bounded by `min`/`max`.
    Integer,
    /// Value: any number, optionally bounded by `min`/`max`.
    Number,
    /// Value: one of the declared `options` strings.
    Select,
}

/// One configurable declared by a scripted driver's `describe()`.
///
/// Flat by design: per-kind constraints are optional fields rather than an
/// internally-tagged enum — serde's internal tagging fought the Lua bridge
/// in step 7, and driver authors write these as flat tables anyway.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct KnobSpec {
    /// Stable key the value is stored and delivered under.
    pub id: String,
    /// Control caption; the form falls back to `id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(rename = "type")]
    pub kind: KnobKind,
    /// Materialized into the frozen config at creation when the client
    /// sends no value. Validated like a submitted value, so a bad
    /// default is the driver author's creation-time error, not a latent
    /// runtime `nil`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// A required knob with no submitted value and no default refuses
    /// creation. Optional knobs may simply be absent — drivers read
    /// missing values as `nil`.
    #[serde(default)]
    pub required: bool,
    /// Lower bound for `integer`/`number` values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    /// Upper bound for `integer`/`number` values.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    /// Legal values for `select`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,
    /// `string` only: request a multi-line control.
    #[serde(default)]
    pub multiline: bool,
}

/// Everything `describe()` may return. A driver without `describe()` gets
/// the default: no metadata, no knobs — exactly the pre-step-10 contract.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct DriverDescription {
    /// Picker display name; the form falls back to the file stem.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// One-line summary shown beside the picker.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub knobs: Vec<KnobSpec>,
}

impl DriverDescription {
    /// Validate submitted knob values against this declaration and
    /// materialize defaults, producing the map frozen onto the weave.
    ///
    /// Refusals name the offending knob so the form can attach the error
    /// to its field. Unknown keys refuse rather than being silently
    /// dropped, per the participant-profiles precedent.
    pub fn validate_config(
        &self,
        submitted: &BTreeMap<String, Value>,
    ) -> Result<BTreeMap<String, Value>, String> {
        for key in submitted.keys() {
            if !self.knobs.iter().any(|spec| &spec.id == key) {
                return Err(format!("unknown knob `{key}` (not declared by describe())"));
            }
        }
        let mut frozen = BTreeMap::new();
        for spec in &self.knobs {
            let value = submitted.get(&spec.id).or(spec.default.as_ref());
            match value {
                Some(value) => {
                    validate_value(spec, value)?;
                    frozen.insert(spec.id.clone(), value.clone());
                }
                None if spec.required => {
                    return Err(format!("knob `{}` is required", spec.id));
                }
                None => {}
            }
        }
        Ok(frozen)
    }
}

fn validate_value(spec: &KnobSpec, value: &Value) -> Result<(), String> {
    let id = &spec.id;
    match spec.kind {
        KnobKind::Model => {
            let Some(map) = value.as_object() else {
                return Err(format!("knob `{id}` expects {{backend, model}}"));
            };
            for field in ["backend", "model"] {
                match map.get(field).and_then(Value::as_str) {
                    Some(s) if !s.is_empty() => {}
                    _ => {
                        return Err(format!("knob `{id}` needs a non-empty `{field}` string"));
                    }
                }
            }
            if let Some(extra) = map.keys().find(|k| *k != "backend" && *k != "model") {
                return Err(format!("knob `{id}` has unexpected field `{extra}`"));
            }
        }
        KnobKind::String => {
            if !value.is_string() {
                return Err(format!("knob `{id}` expects a string"));
            }
        }
        KnobKind::Boolean => {
            if !value.is_boolean() {
                return Err(format!("knob `{id}` expects a boolean"));
            }
        }
        KnobKind::Integer => {
            let Some(n) = value.as_i64() else {
                return Err(format!("knob `{id}` expects a whole number"));
            };
            check_bounds(spec, n as f64)?;
        }
        KnobKind::Number => {
            let Some(n) = value.as_f64() else {
                return Err(format!("knob `{id}` expects a number"));
            };
            check_bounds(spec, n)?;
        }
        KnobKind::Select => {
            if spec.options.is_empty() {
                return Err(format!("knob `{id}` is a select with no options"));
            }
            match value.as_str() {
                Some(s) if spec.options.iter().any(|o| o == s) => {}
                _ => {
                    return Err(format!(
                        "knob `{id}` expects one of: {}",
                        spec.options.join(", ")
                    ));
                }
            }
        }
    }
    Ok(())
}

fn check_bounds(spec: &KnobSpec, n: f64) -> Result<(), String> {
    if let Some(min) = spec.min
        && n < min
    {
        return Err(format!("knob `{}` must be at least {min}", spec.id));
    }
    if let Some(max) = spec.max
        && n > max
    {
        return Err(format!("knob `{}` must be at most {max}", spec.id));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn knob(id: &str, kind: KnobKind) -> KnobSpec {
        KnobSpec {
            id: id.into(),
            label: None,
            kind,
            default: None,
            required: false,
            min: None,
            max: None,
            options: Vec::new(),
            multiline: false,
        }
    }

    fn submitted(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn materializes_defaults_and_skips_absent_optionals() {
        let description = DriverDescription {
            knobs: vec![
                KnobSpec {
                    default: Some(json!(3)),
                    min: Some(1.0),
                    max: Some(5.0),
                    ..knob("rounds", KnobKind::Integer)
                },
                knob("motto", KnobKind::String),
            ],
            ..Default::default()
        };
        let frozen = description.validate_config(&BTreeMap::new()).unwrap();
        assert_eq!(frozen.get("rounds"), Some(&json!(3)));
        assert!(!frozen.contains_key("motto"));
    }

    #[test]
    fn refuses_unknown_required_and_malformed() {
        let description = DriverDescription {
            knobs: vec![
                KnobSpec {
                    required: true,
                    ..knob("voice.model", KnobKind::Model)
                },
                KnobSpec {
                    options: vec!["fast".into(), "careful".into()],
                    ..knob("mode", KnobKind::Select)
                },
            ],
            ..Default::default()
        };
        let ok = submitted(&[(
            "voice.model",
            json!({"backend": "anthropic", "model": "claude-sonnet-5"}),
        )]);
        assert!(description.validate_config(&ok).is_ok());

        let err = description
            .validate_config(&submitted(&[("nope", json!(1))]))
            .unwrap_err();
        assert!(err.contains("unknown knob `nope`"));

        let err = description.validate_config(&BTreeMap::new()).unwrap_err();
        assert!(err.contains("`voice.model` is required"));

        let err = description
            .validate_config(&submitted(&[(
                "voice.model",
                json!({"backend": "anthropic"}),
            )]))
            .unwrap_err();
        assert!(err.contains("non-empty `model`"));

        let mut both = ok.clone();
        both.insert("mode".into(), json!("reckless"));
        let err = description.validate_config(&both).unwrap_err();
        assert!(err.contains("one of: fast, careful"));
    }

    #[test]
    fn bounds_apply_to_defaults_too() {
        let description = DriverDescription {
            knobs: vec![KnobSpec {
                default: Some(json!(9)),
                max: Some(5.0),
                ..knob("rounds", KnobKind::Integer)
            }],
            ..Default::default()
        };
        let err = description.validate_config(&BTreeMap::new()).unwrap_err();
        assert!(err.contains("at most 5"));
    }
}
