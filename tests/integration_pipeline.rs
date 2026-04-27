/// Full pipeline integration tests
/// Requirements:
///   - ONNX models at models/text/model.onnx, models/image/model.onnx
///   - tokenizer at models/text/tokenizer.json
///   - ollama running with llava:7b
///   - Qdrant Cloud reachable (QDRANT_URL and QDRANT_API_KEY in .env)
///   - outfits_test collection created: cargo run --bin setup -- outfits_test
///   - sample image at tests/fixtures/sample.jpg
/// Run with: cargo test --test integration_pipeline -- --nocapture

use fashion_search::processor::model::BufandaAI;
use fashion_search::processor::image::preprocess_image;
use fashion_search::captioner::Captioner;
use fashion_search::db::VectorDB;
use fashion_search::engine::SearchEngine;
use dotenvy::dotenv;

const TEST_COLLECTION: &str = "outfits_test";
const SAMPLE_IMAGE: &str = "tests/fixtures/sample.jpg";

fn make_db() -> VectorDB {
    dotenv().ok();
    VectorDB::new(
        &std::env::var("QDRANT_URL").expect("QDRANT_URL not set"),
        &std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY not set"),
        TEST_COLLECTION,
    )
}

fn make_ai() -> BufandaAI {
    BufandaAI::new(
        "models/text/model.onnx",
        "models/image/model.onnx",
        "models/text/tokenizer.json",
    )
}

// ── db layer ──────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_upsert_and_search_by_text_vector() {
    let mut ai = make_ai();
    let db = make_db();

    let image_vector = ai.encode_image(preprocess_image(SAMPLE_IMAGE));
    let text_vector = ai.encode_text("casual summer outfit with sandals");

    db.upsert_outfit(
        image_vector,
        text_vector,
        SAMPLE_IMAGE,
        "casual summer outfit with sandals",
        "dress",
    )
    .await
    .expect("upsert_outfit failed");

    let results = db
        .search_by_text_vector(ai.encode_text("summer dress"), 20)
        .await
        .expect("search_by_text_vector failed");

    assert!(!results.is_empty(), "Expected at least one result");
    println!("Text search results:");
    for r in &results {
        println!("  score={:.4} | {} | {}", r.score, r.image_url, r.description);
    }
}

#[tokio::test]
async fn test_upsert_and_search_by_image_vector() {
    let mut ai = make_ai();
    let db = make_db();

    let image_vector = ai.encode_image(preprocess_image(SAMPLE_IMAGE));
    let text_vector = ai.encode_text("outfit photo");

    db.upsert_outfit(
        image_vector.clone(),
        text_vector,
        SAMPLE_IMAGE,
        "outfit photo",
        "unknown",
    )
    .await
    .expect("upsert_outfit failed");

    let query_vector = ai.encode_image(preprocess_image(SAMPLE_IMAGE));
    let results = db
        .search_by_image_vector(query_vector, 20)
        .await
        .expect("search_by_image_vector failed");

    assert!(!results.is_empty(), "Expected at least one result");
    assert!(
        results[0].score > 0.99,
        "Top result for same image should have score ~1.0, got {}",
        results[0].score
    );
    println!("Image search top result: score={:.4}", results[0].score);
}

#[tokio::test]
async fn test_search_result_has_required_fields() {
    let mut ai = make_ai();
    let db = make_db();

    db.upsert_outfit(
        ai.encode_image(preprocess_image(SAMPLE_IMAGE)),
        ai.encode_text("test outfit"),
        SAMPLE_IMAGE,
        "test outfit description",
        "test_category",
    )
    .await
    .expect("upsert_outfit failed");

    let results = db
        .search_by_text_vector(ai.encode_text("outfit"), 20)
        .await
        .expect("search failed");

    let top = &results[0];
    assert!(!top.image_url.is_empty(), "image_url should not be empty");
    assert!(top.score > 0.0, "score should be positive");
    println!("Result fields: url='{}' desc='{}' score={:.4}", top.image_url, top.description, top.score);
}

// ── search engine layer ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_engine_search_by_text() {
    let ai = make_ai();
    let db = make_db();
    let mut engine = SearchEngine::new(ai, db);

    let results = engine
        .search_by_text("cute summer dress", 20)
        .await
        .expect("search_by_text failed");

    println!("Engine text search results for 'cute summer dress':");
    for r in &results {
        println!("  score={:.4} | {}", r.score, r.description);
    }
    assert!(results.len() <= 20, "Should return at most 20 results");
}

#[tokio::test]
async fn test_engine_search_by_image() {
    let ai = make_ai();
    let db = make_db();
    let mut engine = SearchEngine::new(ai, db);

    let results = engine
        .search_by_image(SAMPLE_IMAGE, 20)
        .await
        .expect("search_by_image failed");

    println!("Engine image search results:");
    for r in &results {
        println!("  score={:.4} | {}", r.score, r.description);
    }
    assert!(results.len() <= 20, "Should return at most 20 results");
}

// ── captioner + full index pipeline ──────────────────────────────────────────

#[tokio::test]
async fn test_full_index_and_search_pipeline() {
    let mut ai = make_ai();
    let db = make_db();
    let captioner = Captioner::new();

    let caption = captioner
        .caption(SAMPLE_IMAGE)
        .await
        .expect("captioner failed");

    println!("VLM caption: {}", caption);
    assert!(!caption.is_empty());

    let image_vector = ai.encode_image(preprocess_image(SAMPLE_IMAGE));
    let text_vector = ai.encode_text(&caption);

    db.upsert_outfit(image_vector, text_vector, SAMPLE_IMAGE, &caption, "dress")
        .await
        .expect("upsert failed");

    let results = db
        .search_by_text_vector(ai.encode_text("outfit"), 20)
        .await
        .expect("search failed");

    assert!(!results.is_empty(), "Pipeline produced no searchable results");
    println!("Pipeline test top result: score={:.4} | {}", results[0].score, results[0].description);
}
