//! VLM-base image captioning using llava:7b via the ollama HTTP API.
//!
//! Generates natural language descriptions of outfit or clothes photos for use
//! as text embeddings at index time. Not used during search queries.
use serde_json::{json, Value};
use base64::{Engine as _, engine::general_purpose};
use std::fs;

/// HTTP client wrapper for generatin image captions via ollama.
pub struct Captioner {
    client: reqwest::Client,
    model: String,
}

impl Captioner {
    /// Creates a new `Captioner` using `llava:7b` as the VLM.
    ///
    /// Requires ollama to be running locally on port 11434 with `llava:7b` pulled.
    /// The model was chosen for its balance of caption quality and CPU inference
    /// speed on my hardware.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            model: "llava:7b".to_string(),
        }
    }

    /// Generates a natural language description of an outfir or clothes photo
    /// using llava:7b via the ollama HTTP api.
    ///
    /// The image is read from disk, base64-encoded, and sent to ollama along with a fashion-focused
    /// prompt. The response is truncated to 50 words to stay within CLIP's 77-token
    /// context window.
    ///
    /// If ollama returns an empty or missing response, falls back to `"outfit_photo`.
    ///
    ///
    /// # Returns
    ///
    /// A caption string of at most 50 words describing the outfit or the clothesin the photo.
    ///
    /// # Errors
    ///
    /// Returns an error if the image file cannot be read or if the ollama HTTP request fails (e.g.
    /// ollama is not running).
    pub async fn caption(&self, image_path: &str) -> Result<String, Box<dyn std::error::Error>> {
        let image_bytes = fs::read(image_path)?;
        let b64 = general_purpose::STANDARD.encode(&image_bytes);
        let response = self.client
            .post("http://localhost:11434/api/generate")
            .json(&json!({
                "model": self.model,
                "prompt": "Describe this outfit photo in 1-2 sentences. Focus on: clothing items worn, colors, style (casual/formal/sporty), and season suitability. Only describe what you see. Be concise, under 50 words.",
                "images": [b64],
                "stream": false
            }))
            .send()
            .await?;

        let body: Value = response.json().await?;
        let caption = body["response"]
            .as_str()
            .unwrap_or("outfit photo")
            .trim()
            .to_string();

        let truncated = caption
            .split_whitespace()
            .take(50)
            .collect::<Vec<&str>>()
            .join(" ");

        Ok(truncated)
    }
}
