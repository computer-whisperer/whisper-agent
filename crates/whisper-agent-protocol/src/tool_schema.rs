//! Typed tool-input schemas — the shape providers and the UI render
//! against, replacing the per-tool `serde_json::Value` blob we used to
//! pass through verbatim.
//!
//! Tool inputs always reduce to a flat-or-nested record of named
//! parameters. Modeling that record explicitly buys two things:
//!
//! 1. The webui can render a structured manifest (params + types +
//!    descriptions) instead of dumping pretty JSON.
//! 2. Provider adapters get a single conversion point where
//!    Anthropic / OpenAI / Gemini quirks can land — today every adapter
//!    just clones `input_schema: Value` because the typed shape didn't
//!    exist.
//!
//! The MCP wire format and all three provider APIs still consume
//! JSON Schema, so [`ToolSchema::input_schema_value`] reverses the
//! parse and produces the standard `{type:"object", properties:...,
//! required:[...]}` envelope when the request goes out.
//!
//! Anything outside our modeled subset (`oneOf`, `$ref`, multi-type,
//! `pattern`, etc.) lands in [`ParamType::Raw`] and round-trips
//! verbatim — so 3rd-party MCP servers advertising exotic schemas
//! still work, they just don't get the structured render.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// One advertised tool. Mirrors the fields of MCP's `Tool` and the
/// per-provider tool-declaration shapes, but with `params` typed
/// instead of held as a JSON Schema blob.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub params: Vec<ParamSpec>,
    /// Whether this tool runs agent-side (function-call protocol) or
    /// provider-side (built-in tool the model invokes inside the same
    /// request, with the result returned inline). Defaults to
    /// [`ToolKind::Function`] for back-compat with persisted threads
    /// and any third-party caller producing schemas without this field.
    #[serde(default, skip_serializing_if = "ToolKind::is_function")]
    pub kind: ToolKind,
}

/// How a tool is executed. Function tools are dispatched by the agent
/// (built-in handlers, MCP servers, host-env daemons); ProviderBuiltin
/// tools are run inside the provider's API request itself, with the
/// result returned as part of the assistant message — there's no
/// separate `tool_use` → `tool_result` round-trip.
///
/// Today the only ProviderBuiltin we wire up is OpenAI's
/// `image_generation`; the same plumbing extends to `web_search`,
/// `file_search`, `code_interpreter`, etc. when we add them.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    #[default]
    Function,
    ProviderBuiltin,
}

impl ToolKind {
    pub fn is_function(&self) -> bool {
        matches!(self, ToolKind::Function)
    }
}

/// One named parameter on a tool. `required` is denormalized off the
/// JSON Schema `required: [...]` array onto each field so the UI can
/// render without a side lookup.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ParamSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub required: bool,
    pub ty: ParamType,
    /// JSON Schema `default` — kept dynamic because defaults can be
    /// any of the scalar shapes (string/number/bool/array/object).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

/// Modeled subset of JSON Schema types. Anything we can't model lands
/// in [`ParamType::Raw`] and is preserved byte-for-byte through
/// round-trips.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParamType {
    String {
        /// JSON Schema `enum` for string-typed enums (`enum: ["a","b"]`).
        /// Empty means "free-form string". Non-string enums fall to
        /// `Raw` since we can't render them generically.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        enum_values: Vec<String>,
    },
    Integer,
    Number,
    Boolean,
    Array {
        items: Box<ParamType>,
    },
    /// Nested record. Recurses through the same `ParamSpec` shape, so a
    /// renderer can walk objects uniformly with the top-level params.
    Object {
        fields: Vec<ParamSpec>,
    },
    /// Schema shape outside our modeled subset — `oneOf` / `anyOf` /
    /// `$ref` / no-`type` / multi-`type` / etc. The original JSON
    /// Schema fragment is preserved so the provider adapter and MCP
    /// wire still see the exact bytes the tool author wrote, and the
    /// renderer falls back to a JSON view for that one parameter.
    Raw {
        schema: Value,
    },
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl ToolSchema {
    /// Build a typed schema from a wire `(name, description, input_schema)`
    /// triple — the shape MCP servers and persisted v1 conversations
    /// produce. Top-level shapes that aren't `{type: "object", ...}`
    /// collapse into a single root-`Raw` param so the schema still
    /// round-trips, just without structured rendering.
    pub fn from_mcp(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: &Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            params: parse_root_params(input_schema),
            kind: ToolKind::Function,
        }
    }

    /// Re-emit the JSON Schema view of the params — the standard
    /// `{type: "object", properties: {...}, required: [...]}` envelope
    /// that MCP and every provider adapter expects on the wire.
    /// Round-trips lossless for shapes within our modeled subset, and
    /// preserves `Raw` blobs verbatim.
    pub fn input_schema_value(&self) -> Value {
        // Root fallback — `from_mcp` produced a single nameless `Raw`
        // because the input wasn't a `type: object` envelope. Emit the
        // original blob unchanged so a non-conforming 3rd-party schema
        // survives a load+save cycle byte-for-byte.
        if let [single] = self.params.as_slice()
            && single.name.is_empty()
            && let ParamType::Raw { schema } = &single.ty
        {
            return schema.clone();
        }
        params_to_object_schema(&self.params)
    }
}

fn parse_root_params(v: &Value) -> Vec<ParamSpec> {
    let Some(obj) = v.as_object() else {
        return vec![root_fallback(v)];
    };
    if obj.get("type").and_then(|t| t.as_str()) != Some("object") {
        return vec![root_fallback(v)];
    }
    parse_object_fields(obj)
}

fn root_fallback(v: &Value) -> ParamSpec {
    ParamSpec {
        name: String::new(),
        description: String::new(),
        required: false,
        ty: ParamType::Raw { schema: v.clone() },
        default: None,
    }
}

fn parse_object_fields(obj: &Map<String, Value>) -> Vec<ParamSpec> {
    let required: Vec<&str> = obj
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let Some(props) = obj.get("properties").and_then(|p| p.as_object()) else {
        return Vec::new();
    };
    props
        .iter()
        .map(|(k, v)| parse_one_param(k, v, required.contains(&k.as_str())))
        .collect()
}

fn parse_one_param(name: &str, v: &Value, required: bool) -> ParamSpec {
    let obj = v.as_object();
    let description = obj
        .and_then(|o| o.get("description"))
        .and_then(|d| d.as_str())
        .unwrap_or("")
        .to_string();
    let default = obj.and_then(|o| o.get("default")).cloned();
    ParamSpec {
        name: name.to_string(),
        description,
        required,
        ty: parse_param_type(v),
        default,
    }
}

fn parse_param_type(v: &Value) -> ParamType {
    let Some(obj) = v.as_object() else {
        return ParamType::Raw { schema: v.clone() };
    };
    let Some(t) = obj.get("type").and_then(|t| t.as_str()) else {
        return ParamType::Raw { schema: v.clone() };
    };
    match t {
        "string" => {
            let enum_values: Vec<String> = obj
                .get("enum")
                .and_then(|e| e.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            // If the `enum` array existed but contained non-strings,
            // we couldn't model it — fall back to Raw so the values
            // survive verbatim.
            let enum_count = obj
                .get("enum")
                .and_then(|e| e.as_array())
                .map(|a| a.len())
                .unwrap_or(0);
            if enum_count > 0 && enum_values.len() != enum_count {
                return ParamType::Raw { schema: v.clone() };
            }
            ParamType::String { enum_values }
        }
        "integer" => ParamType::Integer,
        "number" => ParamType::Number,
        "boolean" => ParamType::Boolean,
        "array" => {
            let items = obj
                .get("items")
                .map(parse_param_type)
                .unwrap_or(ParamType::Raw {
                    schema: Value::Null,
                });
            ParamType::Array {
                items: Box::new(items),
            }
        }
        "object" => ParamType::Object {
            fields: parse_object_fields(obj),
        },
        _ => ParamType::Raw { schema: v.clone() },
    }
}

fn params_to_object_schema(params: &[ParamSpec]) -> Value {
    let mut properties = Map::new();
    let mut required: Vec<Value> = Vec::new();
    for p in params {
        properties.insert(p.name.clone(), param_to_value(p));
        if p.required {
            required.push(Value::String(p.name.clone()));
        }
    }
    let mut obj = Map::new();
    obj.insert("type".into(), Value::String("object".into()));
    obj.insert("properties".into(), Value::Object(properties));
    if !required.is_empty() {
        obj.insert("required".into(), Value::Array(required));
    }
    Value::Object(obj)
}

fn param_to_value(p: &ParamSpec) -> Value {
    let mut v = type_to_value(&p.ty);
    let Some(map) = v.as_object_mut() else {
        // `Raw` schemas can be any JSON value (e.g. a bare scalar). If
        // the author is decorating it with a description / default, we
        // can't merge — the Raw blob wins as the source of truth.
        return v;
    };
    if !p.description.is_empty() {
        map.insert("description".into(), Value::String(p.description.clone()));
    }
    if let Some(d) = &p.default {
        map.insert("default".into(), d.clone());
    }
    v
}

fn type_to_value(t: &ParamType) -> Value {
    match t {
        ParamType::String { enum_values } => {
            if enum_values.is_empty() {
                json!({ "type": "string" })
            } else {
                let arr: Vec<Value> = enum_values
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                json!({ "type": "string", "enum": arr })
            }
        }
        ParamType::Integer => json!({ "type": "integer" }),
        ParamType::Number => json!({ "type": "number" }),
        ParamType::Boolean => json!({ "type": "boolean" }),
        ParamType::Array { items } => json!({ "type": "array", "items": type_to_value(items) }),
        ParamType::Object { fields } => params_to_object_schema(fields),
        ParamType::Raw { schema } => schema.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_file_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path — absolute, or relative to the workspace root."
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed line to start from. Default 1."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to return. Default: no limit."
                }
            },
            "required": ["path"]
        })
    }

    #[test]
    fn parses_typical_object_schema() {
        let schema = ToolSchema::from_mcp("read_file", "Read the contents", &read_file_schema());
        assert_eq!(schema.name, "read_file");
        assert_eq!(schema.params.len(), 3);
        let path = schema.params.iter().find(|p| p.name == "path").unwrap();
        assert!(path.required);
        assert!(matches!(path.ty, ParamType::String { ref enum_values } if enum_values.is_empty()));
        let offset = schema.params.iter().find(|p| p.name == "offset").unwrap();
        assert!(!offset.required);
        assert!(matches!(offset.ty, ParamType::Integer));
    }

    #[test]
    fn round_trips_object_schema() {
        let original = read_file_schema();
        let typed = ToolSchema::from_mcp("read_file", "desc", &original);
        let back = typed.input_schema_value();
        // Property order isn't guaranteed across the round-trip
        // (serde_json::Map preserves insertion order, but JSON Schema
        // semantics don't care). Compare as parsed values.
        assert_eq!(back["type"], original["type"]);
        assert_eq!(back["properties"], original["properties"]);
        assert_eq!(back["required"], original["required"]);
    }

    #[test]
    fn handles_string_enum() {
        let v = json!({
            "type": "object",
            "properties": {
                "mode": { "type": "string", "enum": ["sync", "async"] }
            },
            "required": ["mode"]
        });
        let s = ToolSchema::from_mcp("t", "", &v);
        let mode = &s.params[0];
        match &mode.ty {
            ParamType::String { enum_values } => {
                assert_eq!(enum_values, &["sync".to_string(), "async".to_string()]);
            }
            other => panic!("expected String, got {other:?}"),
        }
        let back = s.input_schema_value();
        assert_eq!(back["properties"]["mode"]["enum"], json!(["sync", "async"]));
    }

    #[test]
    fn handles_array_of_string() {
        let v = json!({
            "type": "object",
            "properties": {
                "files": { "type": "array", "items": { "type": "string" } }
            }
        });
        let s = ToolSchema::from_mcp("t", "", &v);
        let files = &s.params[0];
        match &files.ty {
            ParamType::Array { items } => {
                assert!(
                    matches!(**items, ParamType::String { ref enum_values } if enum_values.is_empty())
                );
            }
            other => panic!("expected Array, got {other:?}"),
        }
    }

    #[test]
    fn handles_nested_object() {
        let v = json!({
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "host": { "type": "string", "description": "hostname" },
                        "port": { "type": "integer" }
                    },
                    "required": ["host"]
                }
            }
        });
        let s = ToolSchema::from_mcp("t", "", &v);
        let config = &s.params[0];
        match &config.ty {
            ParamType::Object { fields } => {
                assert_eq!(fields.len(), 2);
                let host = fields.iter().find(|p| p.name == "host").unwrap();
                assert_eq!(host.description, "hostname");
                assert!(host.required);
            }
            other => panic!("expected Object, got {other:?}"),
        }
        // Round-trip preserves nested structure.
        let back = s.input_schema_value();
        assert_eq!(
            back["properties"]["config"]["properties"]["host"]["type"],
            "string"
        );
        assert_eq!(back["properties"]["config"]["required"], json!(["host"]));
    }

    #[test]
    fn falls_back_to_raw_for_unmodeled_shape() {
        // `oneOf` is outside our modeled subset.
        let v = json!({
            "type": "object",
            "properties": {
                "weird": { "oneOf": [{ "type": "string" }, { "type": "integer" }] }
            }
        });
        let s = ToolSchema::from_mcp("t", "", &v);
        let weird = &s.params[0];
        assert!(matches!(weird.ty, ParamType::Raw { .. }));
        let back = s.input_schema_value();
        assert_eq!(
            back["properties"]["weird"],
            json!({ "oneOf": [{ "type": "string" }, { "type": "integer" }] })
        );
    }

    #[test]
    fn root_fallback_for_non_object_schema() {
        // Pathological: top-level isn't an object envelope. The whole
        // blob gets wrapped in a single root-Raw param so a load+save
        // emits the original bytes.
        let v = json!("not even an object");
        let s = ToolSchema::from_mcp("t", "", &v);
        assert_eq!(s.params.len(), 1);
        assert_eq!(s.params[0].name, "");
        assert!(matches!(s.params[0].ty, ParamType::Raw { .. }));
        assert_eq!(s.input_schema_value(), v);
    }

    #[test]
    fn object_with_no_properties_is_empty_param_list() {
        let v = json!({ "type": "object", "properties": {} });
        let s = ToolSchema::from_mcp("t", "", &v);
        assert!(s.params.is_empty());
        let back = s.input_schema_value();
        assert_eq!(back["type"], "object");
        assert_eq!(back["properties"], json!({}));
    }

    #[test]
    fn defaults_round_trip() {
        let v = json!({
            "type": "object",
            "properties": {
                "timeout": { "type": "integer", "default": 5000 },
                "verbose": { "type": "boolean", "default": false }
            }
        });
        let s = ToolSchema::from_mcp("t", "", &v);
        let timeout = s.params.iter().find(|p| p.name == "timeout").unwrap();
        assert_eq!(timeout.default, Some(json!(5000)));
        let back = s.input_schema_value();
        assert_eq!(back["properties"]["timeout"]["default"], json!(5000));
        assert_eq!(back["properties"]["verbose"]["default"], json!(false));
    }

    #[test]
    fn typed_form_serde_round_trip() {
        // Bincode-shaped (the persisted form): build a typed schema,
        // serialize via serde to JSON, deserialize back, confirm equal.
        let typed = ToolSchema::from_mcp("read_file", "Read", &read_file_schema());
        let bytes = serde_json::to_vec(&typed).unwrap();
        let again: ToolSchema = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(typed, again);
    }
}
