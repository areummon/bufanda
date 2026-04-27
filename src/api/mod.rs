//! Axum HTTP API layer for the search engine.
//!
//! Exposes three endpoints:
//! - `Get /health`         - liveness check
//! - `POST /search/text`   - text query search
//! - `POST /search/image`  - image upload search
//! - `GET /image`          - serves dataset images by path
use axum::{
    Router, extract::{Multipart, State}, http::StatusCode, response::{IntoResponse, Json}, routing::{get, post}
};
use axum::extract::Query;
use tower_http::services::ServeDir;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tokio::sync::Mutex;

use crate::engine::SearchEngine;

/// Shared application state passed to every request handler.
///
/// Wraps the [`SearchEngine`] in `Arc<Mutex<>>` so it can be safely shared 
/// across concurrent requests. The `Mutex` is needed because ONNX inference
/// sessions are not thread-safe.
pub type AppState = Arc<Mutex<SearchEngine>>;

#[derive(Deserialize)]
pub struct TextQuery {
    pub query: String,
}

#[derive(Serialize)]
pub struct SearchResultResponse {
    pub image_url: String,
    pub description: String,
    pub score: f32,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
}

/// Builds the Axum router with all routes and shared state attached.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/search/text", post(search_by_text))
        .route("/search/image", post(search_by_image))
        .route("/image", get(serve_image))
        .fallback_service(ServeDir::new("static"))
        .with_state(state)
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
    })
}

async fn serve_image(
    Query(params): Query<HashMap<String, String>>,
) -> impl axum::response::IntoResponse {
    let path = params.get("path").cloned().unwrap_or_default();
    match tokio::fs::read(&path).await {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "image/jpeg")],
            bytes,
        ).into_response(),
        Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
}

async fn search_by_text(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
    Json(payload): Json<TextQuery>,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    if payload.query.trim().is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Query cannot be empty".to_string()));
    }

    let limit = params.get("limit")
        .and_then(|l| l.parse::<u64>().ok())
        .unwrap_or(20)
        .min(100);

    let mut engine = state.lock().await;
    let results = engine
        .search_by_text(&payload.query, limit)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(|r| SearchResultResponse {
                image_url: r.image_url,
                description: r.description,
                score: r.score,
            })
            .collect(),
    }))
}

async fn search_by_image(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
    mut multipart: Multipart,
) -> Result<Json<SearchResponse>, (StatusCode, String)> {
    let limit = params.get("limit")
        .and_then(|l| l.parse::<u64>().ok())
        .unwrap_or(20)
        .min(100);

    let field = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
        .ok_or((StatusCode::BAD_REQUEST, "No image field in request".to_string()))?;

    let bytes = field
        .bytes()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Write the bytes to a temporary file, necessary for the program.
    let tmp_path = format!("/tmp/search_upload_{}.jpg", uuid::Uuid::new_v4());
    tokio::fs::write(&tmp_path, &bytes)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut engine = state.lock().await;
    let results = engine
        .search_by_image(&tmp_path, limit)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let _ = tokio::fs::remove_file(&tmp_path).await;

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(|r| SearchResultResponse {
                image_url: r.image_url,
                description: r.description,
                score: r.score,
            })
            .collect(),
    }))
}

