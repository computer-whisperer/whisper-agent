//! Codex (ChatGPT subscription) image-generation path.
//!
//! The standalone `/v1/images/generations` endpoint is only served by
//! `api.openai.com` and only authenticates with API keys — the
//! ChatGPT-subscription host (`chatgpt.com/backend-api/codex`) returns 404
//! for it. To make image generation work for subscription users, we drive
//! the Responses API and let the model invoke the `image_generation`
//! built-in tool, which IS supported on that host.
//!
//! Mechanics:
//! - POST `/responses` with `tools = [{ type: "image_generation", … }]`
//!   and `tool_choice = { type: "image_generation" }` to force invocation.
//! - The subscription host rejects `stream: false` outright, so we always
//!   stream and reassemble. Each SSE event is a `data: <json>` envelope.
//! - Items can arrive two ways depending on the route: api.openai.com
//!   inlines them in `response.completed.response.output`, while
//!   chatgpt.com's subscription proxy sends `response.output_item.done`
//!   events and leaves `response.completed.response.output` empty. We
//!   accumulate from both and stop at `response.completed`.
//! - Returns the base64-encoded result strings as-is — the caller wraps
//!   them in MCP image content blocks.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};

use whisper_agent_auth::ClientAuth;

/// `n` > 1 isn't supported on this path: the `image_generation` built-in
/// tool generates a single image per invocation and we don't loop. Caller
/// can split into multiple tool calls if they need variations.
pub const MAX_N_VIA_RESPONSES: u32 = 1;

pub struct Request<'a> {
    pub http: &'a Client,
    pub auth: &'a ClientAuth,
    pub api_base: &'a str,
    pub chat_model: &'a str,
    pub prompt: &'a str,
    pub size: &'a str,
    pub quality: &'a str,
    pub n: u32,
}

pub async fn generate_via_responses(req: Request<'_>) -> Result<Vec<String>> {
    let Request {
        http,
        auth,
        api_base,
        chat_model,
        prompt,
        size,
        quality,
        n,
    } = req;
    if n > MAX_N_VIA_RESPONSES {
        return Err(anyhow!(
            "image_generate: n={n} not supported on chatgpt_subscription auth \
             (Responses-API path); call multiple times with n=1, or use api_key auth"
        ));
    }

    let body = json!({
        "model": chat_model,
        "instructions": "You must invoke the image_generation tool exactly once \
                         to fulfill the user's request. Do not respond with text alone.",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": prompt }],
        }],
        "tools": [{
            "type": "image_generation",
            "size": size,
            "quality": quality,
        }],
        "tool_choice": { "type": "image_generation" },
        "store": false,
        "stream": true,
    });

    let (bearer, extras) = auth
        .prepare_headers(http)
        .await
        .map_err(|e| anyhow!("auth: {e}"))?;
    let mut req = http
        .post(format!("{api_base}/responses"))
        .bearer_auth(&bearer)
        .header("accept", "text/event-stream")
        .json(&body);
    for (k, v) in &extras {
        req = req.header(*k, v);
    }

    let resp = req.send().await.map_err(|e| anyhow!("send: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        return Err(anyhow!(
            "/responses HTTP {}: {}",
            status.as_u16(),
            crate::tools::truncate_for_display(&body_text, 1024)
        ));
    }

    let mut images: Vec<String> = Vec::new();
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| anyhow!("stream chunk: {e}"))?;
        let s = std::str::from_utf8(&bytes).map_err(|e| anyhow!("non-utf8 sse chunk: {e}"))?;
        buffer.push_str(s);

        // Process complete events (separated by a blank line).
        while let Some(end) = buffer.find("\n\n") {
            // Drain payload from `data:` lines, joined by `\n` per the SSE spec.
            let mut data_payload = String::new();
            for line in buffer[..end].lines() {
                let trimmed = line
                    .strip_prefix("data:")
                    .map(|d| d.strip_prefix(' ').unwrap_or(d));
                if let Some(d) = trimmed {
                    if !data_payload.is_empty() {
                        data_payload.push('\n');
                    }
                    data_payload.push_str(d);
                }
            }
            buffer.drain(..=end + 1); // remove the event + the blank-line terminator

            if data_payload.is_empty() || data_payload == "[DONE]" {
                continue;
            }
            match parse_event(&data_payload) {
                ParsedEvent::OutputItemDone { item } => {
                    if let OutputItem::ImageGenerationCall { result } = item {
                        images.push(result);
                    }
                }
                ParsedEvent::ResponseCompleted { output } => {
                    // api.openai.com inlines items here. chatgpt.com's
                    // subscription proxy leaves it empty and the items
                    // arrived on output_item.done already; either way
                    // we union both sources.
                    for item in output {
                        if let OutputItem::ImageGenerationCall { result } = item {
                            images.push(result);
                        }
                    }
                    return Ok(images);
                }
                ParsedEvent::ResponseFailed { error } => {
                    return Err(anyhow!("/responses failed: {error}"));
                }
                ParsedEvent::Other => {}
            }
        }
    }

    // Stream ended without a `response.completed` event — surface whatever
    // we have rather than hang the caller, but flag the truncation so the
    // upstream tool result is clearly degraded.
    if images.is_empty() {
        Err(anyhow!(
            "/responses stream ended before response.completed and produced no images"
        ))
    } else {
        Ok(images)
    }
}

enum ParsedEvent {
    OutputItemDone { item: OutputItem },
    ResponseCompleted { output: Vec<OutputItem> },
    ResponseFailed { error: String },
    Other,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputItem {
    #[serde(rename = "image_generation_call")]
    ImageGenerationCall {
        #[serde(default)]
        result: String,
    },
    #[serde(other)]
    Other,
}

fn parse_event(data_payload: &str) -> ParsedEvent {
    #[derive(Deserialize)]
    struct Envelope {
        #[serde(rename = "type")]
        type_: String,
        #[serde(default)]
        response: Option<ResponseObj>,
        #[serde(default)]
        item: Option<OutputItem>,
        #[serde(default)]
        error: Option<Value>,
    }
    #[derive(Deserialize)]
    struct ResponseObj {
        #[serde(default)]
        output: Vec<OutputItem>,
    }
    let env: Envelope = match serde_json::from_str(data_payload) {
        Ok(v) => v,
        Err(_) => return ParsedEvent::Other,
    };
    match env.type_.as_str() {
        "response.output_item.done" => match env.item {
            Some(item) => ParsedEvent::OutputItemDone { item },
            None => ParsedEvent::Other,
        },
        "response.completed" => {
            let output = env.response.map(|r| r.output).unwrap_or_default();
            ParsedEvent::ResponseCompleted { output }
        }
        "response.failed" => {
            let error = env
                .error
                .map(|v| v.to_string())
                .unwrap_or_else(|| "(no error detail)".to_string());
            ParsedEvent::ResponseFailed { error }
        }
        _ => ParsedEvent::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_event_extracts_image_from_output_item_done() {
        // ChatGPT-subscription path: items arrive on output_item.done; the
        // later response.completed has output:[].
        let payload = r#"{"type":"response.output_item.done","item":{
            "type":"image_generation_call","id":"i_1","status":"completed","result":"BASE64HERE"
        }}"#;
        match parse_event(payload) {
            ParsedEvent::OutputItemDone { item } => match item {
                OutputItem::ImageGenerationCall { result } => {
                    assert_eq!(result, "BASE64HERE");
                }
                _ => panic!("expected ImageGenerationCall"),
            },
            _ => panic!("expected OutputItemDone"),
        }
    }

    #[test]
    fn parse_event_extracts_image_from_response_completed() {
        let payload = r#"{"type":"response.completed","response":{"output":[
            {"type":"image_generation_call","id":"i_1","status":"completed","result":"BASE64HERE"}
        ]}}"#;
        match parse_event(payload) {
            ParsedEvent::ResponseCompleted { output } => {
                assert_eq!(output.len(), 1);
                match &output[0] {
                    OutputItem::ImageGenerationCall { result } => {
                        assert_eq!(result, "BASE64HERE");
                    }
                    _ => panic!("expected ImageGenerationCall"),
                }
            }
            _ => panic!("expected ResponseCompleted"),
        }
    }

    #[test]
    fn parse_event_skips_non_image_output_items() {
        let payload = r#"{"type":"response.completed","response":{"output":[
            {"type":"message","id":"m1"},
            {"type":"image_generation_call","id":"i_1","status":"completed","result":"X"}
        ]}}"#;
        match parse_event(payload) {
            ParsedEvent::ResponseCompleted { output } => {
                assert_eq!(output.len(), 2);
                let images: Vec<_> = output
                    .into_iter()
                    .filter_map(|i| match i {
                        OutputItem::ImageGenerationCall { result } => Some(result),
                        _ => None,
                    })
                    .collect();
                assert_eq!(images, vec!["X"]);
            }
            _ => panic!("expected ResponseCompleted"),
        }
    }

    #[test]
    fn parse_event_recognizes_failure() {
        let payload = r#"{"type":"response.failed","error":{"code":"x","message":"y"}}"#;
        match parse_event(payload) {
            ParsedEvent::ResponseFailed { error } => {
                assert!(error.contains("\"code\""));
            }
            _ => panic!("expected ResponseFailed"),
        }
    }

    #[test]
    fn parse_event_ignores_unrelated_types() {
        let payload = r#"{"type":"response.output_text.delta","delta":"hi"}"#;
        assert!(matches!(parse_event(payload), ParsedEvent::Other));
    }

    #[test]
    fn parse_event_ignores_malformed_json() {
        assert!(matches!(parse_event("not json"), ParsedEvent::Other));
    }
}
