//! The single `image_generate` tool. Hits the OpenAI Images API and returns
//! the generated image(s) as MCP image content blocks.

use std::sync::Arc;

use arc_swap::ArcSwap;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::warn;

use whisper_agent_mcp_proto::{CallToolResult, ContentBlock, Tool, ToolAnnotations};

use crate::config::Resolved;

#[derive(Debug, thiserror::Error)]
pub enum ToolDispatchError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
}

/// OpenAI caps `n` at 10 per /v1/images/generations call; we mirror that
/// rather than relying on the upstream error.
const MAX_N: u32 = 10;

/// Daemon state shared across all in-flight tool calls. `resolved` is
/// hot-swapped wholesale on config-file change; `default_size`/`default_quality`
/// are daemon-level CLI args and never change for the lifetime of the process.
pub struct ImageGenConfig {
    pub http: Client,
    pub resolved: Arc<ArcSwap<Resolved>>,
    pub default_size: String,
    pub default_quality: String,
}

pub fn descriptors() -> Vec<Tool> {
    vec![image_generate_descriptor()]
}

pub async fn call(
    cfg: &Arc<ImageGenConfig>,
    name: &str,
    args: Value,
) -> Result<CallToolResult, ToolDispatchError> {
    match name {
        "image_generate" => Ok(image_generate(cfg, args).await),
        _ => Err(ToolDispatchError::UnknownTool(name.to_string())),
    }
}

// ---------- image_generate ----------

fn image_generate_descriptor() -> Tool {
    Tool {
        name: "image_generate".into(),
        description: "Generate one or more images from a text prompt using OpenAI's \
                      image-generation API (gpt-image-2 by default). Images are returned \
                      inline as PNG content blocks the model can see and reference. Use \
                      this when the user asks for a picture, diagram, mockup, or any other \
                      visual output. Cost scales with `quality` and `size` — defaults are \
                      `auto` for both, which lets the API pick a sensible balance. Pass \
                      `n` > 1 to get multiple variations of the same prompt in one call."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Description of the image to generate. Be specific about \
                                    subject, style, composition, and any text that should appear."
                },
                "n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Number of images to generate (1-10). Defaults to 1."
                },
                "size": {
                    "type": "string",
                    "description": "Image dimensions, e.g. `1024x1024`, `1536x1024`, \
                                    `1024x1536`, or `auto` to let the model choose. \
                                    Defaults to the daemon's configured default."
                },
                "quality": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "auto"],
                    "description": "Generation quality. Higher quality costs more and takes \
                                    longer. Defaults to the daemon's configured default."
                },
                "model": {
                    "type": "string",
                    "description": "Override the default model (e.g. `gpt-image-2`, \
                                    `gpt-image-1`). Most callers should leave this unset."
                }
            },
            "required": ["prompt"]
        }),
        annotations: Some(ToolAnnotations {
            title: Some("Generate an image".into()),
            read_only_hint: Some(false),
            destructive_hint: Some(false),
            // Same prompt produces different images each call (generation is stochastic).
            idempotent_hint: Some(false),
            open_world_hint: Some(false),
        }),
    }
}

#[derive(Deserialize)]
struct ImageGenerateArgs {
    prompt: String,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    quality: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiImagesResponse {
    #[serde(default)]
    data: Vec<OpenAiImageDatum>,
}

#[derive(Deserialize)]
struct OpenAiImageDatum {
    /// Base64-encoded PNG. Always present for `gpt-image-*` models; for
    /// `dall-e-*` models the API defaults to `url` instead — we don't
    /// special-case that path and surface a clear error if neither field
    /// is set.
    #[serde(default)]
    b64_json: Option<String>,
}

async fn image_generate(cfg: &Arc<ImageGenConfig>, args: Value) -> CallToolResult {
    let parsed: ImageGenerateArgs = match serde_json::from_value(args) {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("invalid arguments: {e}")),
    };

    if parsed.prompt.trim().is_empty() {
        return CallToolResult::error_text("prompt must not be empty");
    }

    let n = parsed.n.unwrap_or(1).clamp(1, MAX_N);
    // Snapshot the live config under arc-swap so an in-flight reload can't
    // pull the rug out from under us mid-request. Subsequent `prepare_headers`
    // and HTTP send all see the same auth + base + model.
    let resolved = cfg.resolved.load_full();
    let model = parsed.model.as_deref().unwrap_or(&resolved.default_model);
    let size = parsed.size.as_deref().unwrap_or(&cfg.default_size);
    let quality = parsed.quality.as_deref().unwrap_or(&cfg.default_quality);

    let body = json!({
        "model": model,
        "prompt": parsed.prompt,
        "n": n,
        "size": size,
        "quality": quality,
    });

    let (bearer, extra_headers) = match resolved.auth.prepare_headers(&cfg.http).await {
        Ok(v) => v,
        Err(e) => return CallToolResult::error_text(format!("image_generate auth: {e}")),
    };

    let url = format!("{}/images/generations", resolved.api_base);
    let mut req = cfg.http.post(&url).bearer_auth(&bearer).json(&body);
    for (k, v) in &extra_headers {
        req = req.header(*k, v);
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => return CallToolResult::error_text(format!("image_generate request: {e}")),
    };

    let status = resp.status();
    if !status.is_success() {
        let body_text = resp
            .text()
            .await
            .unwrap_or_else(|e| format!("(failed to read body: {e})"));
        warn!(%status, body = %body_text, "openai images api error");
        return CallToolResult::error_text(format!(
            "image_generate failed: HTTP {} — {}",
            status.as_u16(),
            truncate_for_display(&body_text, 1024)
        ));
    }

    let parsed_body: OpenAiImagesResponse = match resp.json().await {
        Ok(v) => v,
        Err(e) => {
            return CallToolResult::error_text(format!("image_generate decode response: {e}"));
        }
    };

    if parsed_body.data.is_empty() {
        return CallToolResult::error_text("image_generate: response contained no images");
    }

    let mut content: Vec<ContentBlock> = Vec::with_capacity(parsed_body.data.len());
    for (idx, datum) in parsed_body.data.into_iter().enumerate() {
        let Some(b64) = datum.b64_json else {
            return CallToolResult::error_text(format!(
                "image_generate: image {idx} had no `b64_json` field — \
                 this MCP server only supports models that return inline base64 \
                 (gpt-image-* family); got model `{model}`"
            ));
        };
        content.push(ContentBlock::Image {
            data: b64,
            mime_type: "image/png".into(),
        });
    }

    CallToolResult {
        content,
        is_error: None,
    }
}

/// Truncate a UTF-8 string to at most `max_bytes` for use in error messages.
/// Walks back to a char boundary; appends an ellipsis if anything was dropped.
fn truncate_for_display(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_returns_full() {
        assert_eq!(truncate_for_display("hello", 100), "hello");
    }

    #[test]
    fn truncate_long_appends_ellipsis() {
        let s = "a".repeat(200);
        let out = truncate_for_display(&s, 50);
        assert_eq!(out.len(), 50 + '…'.len_utf8());
        assert!(out.ends_with('…'));
    }

    #[test]
    fn truncate_walks_to_char_boundary() {
        let s = "ééééé"; // each é is 2 bytes; total 10 bytes
        let out = truncate_for_display(s, 5);
        // Walked back to byte 4 (two complete é's), appended '…'.
        assert_eq!(out, "éé…");
    }

    #[test]
    fn descriptor_input_schema_is_object_with_prompt() {
        let d = image_generate_descriptor();
        let schema = &d.input_schema;
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["prompt"].is_object());
        assert_eq!(schema["required"][0], "prompt");
    }
}
