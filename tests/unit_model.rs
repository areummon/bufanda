/// Tests for BufandaAI encoding
/// Requirements: ONNX models at models/text/model.onnx, models/image/model.onnx
///               tokenizer at models/text/tokenizer.json
///               sample image at tests/fixtures/sample.jpg
/// Run with: cargo test --test unit_model

use fashion_search::processor::model::BufandaAI;
use fashion_search::processor::image::preprocess_image;

fn make_ai() -> BufandaAI {
    BufandaAI::new(
        "models/text/model.onnx",
        "models/image/model.onnx",
        "models/text/tokenizer.json",
    )
}

// ── normalize math ────────────────────────────────────────────────────────────
// These don't need models, they test the math directly via encode_text output

#[test]
fn test_text_vector_is_unit_length() {
    let mut ai = make_ai();
    let vec = ai.encode_text("a red dress");
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Text vector norm is {} (expected ~1.0)",
        norm
    );
}

#[test]
fn test_text_vector_is_512_dims() {
    let mut ai = make_ai();
    let vec = ai.encode_text("casual summer outfit");
    assert_eq!(vec.len(), 512, "Expected 512-dim vector, got {}", vec.len());
}

#[test]
fn test_text_vector_no_nan_or_inf() {
    let mut ai = make_ai();
    let vec = ai.encode_text("white sneakers and jeans");
    for val in &vec {
        assert!(!val.is_nan(), "Text vector contains NaN");
        assert!(!val.is_infinite(), "Text vector contains Inf");
    }
}

#[test]
fn test_image_vector_is_unit_length() {
    let mut ai = make_ai();
    let tensor = preprocess_image("tests/fixtures/sample.jpg");
    let vec = ai.encode_image(tensor);
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Image vector norm is {} (expected ~1.0)",
        norm
    );
}

#[test]
fn test_image_vector_is_512_dims() {
    let mut ai = make_ai();
    let tensor = preprocess_image("tests/fixtures/sample.jpg");
    let vec = ai.encode_image(tensor);
    assert_eq!(vec.len(), 512, "Expected 512-dim vector, got {}", vec.len());
}

#[test]
fn test_image_vector_no_nan_or_inf() {
    let mut ai = make_ai();
    let tensor = preprocess_image("tests/fixtures/sample.jpg");
    let vec = ai.encode_image(tensor);
    for val in &vec {
        assert!(!val.is_nan(), "Image vector contains NaN");
        assert!(!val.is_infinite(), "Image vector contains Inf");
    }
}

// ── semantic sanity ───────────────────────────────────────────────────────────

#[test]
fn test_similar_queries_produce_close_vectors() {
    let mut ai = make_ai();
    let v1 = ai.encode_text("summer dress");
    let v2 = ai.encode_text("dress for summer");

    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    // Both are unit vectors so dot product == cosine similarity
    // Similar phrases should score > 0.8
    assert!(
        dot > 0.8,
        "Similar queries cosine similarity is {} (expected > 0.8)",
        dot
    );
}

#[test]
fn test_different_queries_produce_different_vectors() {
    let mut ai = make_ai();
    let v1 = ai.encode_text("red evening gown");
    let v2 = ai.encode_text("hiking boots and waterproof jacket");

    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    // Very different concepts should score lower than similar ones
    assert!(
        dot < 0.95,
        "Unrelated queries are suspiciously similar: cosine = {}",
        dot
    );
}

#[test]
fn test_empty_string_does_not_panic() {
    let mut ai = make_ai();
    // Should not panic — may return a valid or zeroed vector
    let vec = ai.encode_text("");
    assert_eq!(vec.len(), 512);
}
