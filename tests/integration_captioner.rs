/// Integration tests for the Captioner
/// Requirements: ollama running on localhost:11434 with moondream pulled
///               sample image at tests/fixtures/sample.jpg
/// Run with: cargo test --test integration_captioner -- --nocapture

use fashion_search::captioner::Captioner;

#[tokio::test]
async fn test_caption_returns_non_empty_string() {
    let captioner = Captioner::new();
    let result = captioner.caption("tests/fixtures/sample.jpg").await;

    assert!(result.is_ok(), "Caption failed with error: {:?}", result.err());
    let caption = result.unwrap();
    assert!(!caption.is_empty(), "Caption should not be empty");
    println!("Caption: {}", caption);
}

#[tokio::test]
async fn test_caption_is_readable_text() {
    let captioner = Captioner::new();
    let caption = captioner
        .caption("tests/fixtures/sample.jpg")
        .await
        .unwrap();

    // Should contain at least a few words
    let word_count = caption.split_whitespace().count();
    assert!(
        word_count >= 5,
        "Caption is too short ({} words): '{}'",
        word_count,
        caption
    );
}

#[tokio::test]
async fn test_caption_bad_path_returns_error() {
    let captioner = Captioner::new();
    let result = captioner.caption("nonexistent/path/image.jpg").await;
    assert!(result.is_err(), "Expected error for nonexistent file");
}
