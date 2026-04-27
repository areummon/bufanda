/// End-to-end test — indexes 10 user images then queries the live API
///
/// Requirements:
///   - Server running:       cargo run
///   - Indexer run first:    cargo run --bin indexer
///   - Or use this script which does a mini-index then queries
///
/// Run with: cargo test --test e2e -- --nocapture --test-threads=1

use fashion_search::processor::model::BufandaAI;
use fashion_search::processor::image::preprocess_image;
use fashion_search::captioner::Captioner;
use fashion_search::db::VectorDB;
use fashion_search::engine::SearchEngine;
use dotenvy::dotenv;
use std::fs;
use serde_json::Value;

const SMALL_BATCH: usize = 10;
const IMAGE_DIR: &str = "./data/validation/image";
const ANNO_DIR: &str = "./data/validation/annos";
const SAMPLE_IMAGE: &str = "tests/fixtures/sample.jpg";

fn make_ai() -> BufandaAI {
    BufandaAI::new(
        "models/text/model.onnx",
        "models/image/model.onnx",
        "models/text/tokenizer.json",
    )
}

fn make_db(collection: &str) -> VectorDB {
    dotenv().ok();
    VectorDB::new(
        &std::env::var("QDRANT_URL").expect("QDRANT_URL not set"),
        &std::env::var("QDRANT_API_KEY").expect("QDRANT_API_KEY not set"),
        collection,
    )
}

// ── step 1: index a small batch ───────────────────────────────────────────────

#[tokio::test]
async fn test_e2e_index_small_batch() {
    let mut ai = make_ai();
    let db = make_db("outfits_test");
    let captioner = Captioner::new();

    let entries: Vec<_> = fs::read_dir(IMAGE_DIR)
        .expect("Cannot read image dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_file()
                && e.path().extension().map_or(false, |ext| ext == "jpg")
        })
        .take(SMALL_BATCH * 10)
        .collect();

    let mut indexed = 0;

    for entry in entries {
        if indexed >= SMALL_BATCH {
            break;
        }

        let img_path = entry.path();
        let file_stem = img_path.file_stem().unwrap().to_str().unwrap();
        let json_path = format!("{}/{}.json", ANNO_DIR, file_stem);
        let path_str = img_path.to_str().unwrap();

        let json_content = match fs::read_to_string(&json_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let parsed: Value = match serde_json::from_str(&json_content) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let source = parsed.get("source").and_then(|s| s.as_str()).unwrap_or("");
        if source != "user" {
            continue;
        }

        let category = parsed
            .as_object()
            .map(|obj| {
                obj.iter()
                    .filter(|(k, _)| k.starts_with("item"))
                    .filter_map(|(_, v)| {
                        v.get("category_name").and_then(|c| c.as_str()).map(String::from)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        let description = captioner
            .caption(path_str)
            .await
            .unwrap_or_else(|_| category.clone());

        let image_vector = ai.encode_image(preprocess_image(path_str));
        let text_vector = ai.encode_text(&description);

        db.upsert_outfit(image_vector, text_vector, path_str, &description, &category)
            .await
            .expect("upsert failed");

        println!("✅ [{}/{}] {} → {}", indexed + 1, SMALL_BATCH, file_stem, description);
        indexed += 1;
    }

    assert!(indexed > 0, "Could not find any user images to index");
    println!("\nSmall batch indexing complete. {} images indexed.", indexed);
}

// ── step 2: query the indexed data directly (no HTTP) ─────────────────────────

#[tokio::test]
async fn test_e2e_text_search_returns_relevant_results() {
    let mut ai = make_ai();
    let db = make_db("outfits_test");

    let queries = vec![
        "casual summer outfit",
        "dress",
        "trousers and top",
    ];

    for query in queries {
        let vector = ai.encode_text(query);
        let results = db
            .search_by_text_vector(vector, 20)
            .await
            .expect("search failed");

        println!("\nQuery: '{}'", query);
        assert!(!results.is_empty(), "No results for query '{}'", query);

        for r in &results {
            println!("  score={:.4} | {} | {}", r.score, r.image_url, r.description);
        }

        assert!(
            results[0].score > 0.1,
            "Top result score too low ({}) for query '{}'",
            results[0].score,
            query
        );
    }
}

#[tokio::test]
async fn test_e2e_image_search_returns_results() {
    let mut ai = make_ai();
    let db = make_db("outfits_test");

    let tensor = preprocess_image(SAMPLE_IMAGE);
    let vector = ai.encode_image(tensor);

    let results = db
        .search_by_image_vector(vector, 20)
        .await
        .expect("image search failed");

    println!("\nImage search results:");
    assert!(!results.is_empty(), "No results for image query");

    for r in &results {
        println!("  score={:.4} | {}", r.score, r.description);
    }
}

// ── step 3: engine layer ──────────────────────────────────────────────────────

#[tokio::test]
async fn test_e2e_engine_text_search() {
    let ai = make_ai();
    let db = make_db("outfits_test");
    let mut engine = SearchEngine::new(ai, db);

    let results = engine
        .search_by_text("casual summer dress", 20)
        .await
        .expect("engine search_by_text failed");

    println!("\nEngine text search results:");
    for r in &results {
        println!("  score={:.4} | {}", r.score, r.description);
    }
    assert!(results.len() <= 20);
}

#[tokio::test]
async fn test_e2e_engine_image_search() {
    let ai = make_ai();
    let db = make_db("outfits_test");
    let mut engine = SearchEngine::new(ai, db);

    let results = engine
        .search_by_image(SAMPLE_IMAGE, 20)
        .await
        .expect("engine search_by_image failed");

    println!("\nEngine image search results:");
    for r in &results {
        println!("  score={:.4} | {}", r.score, r.description);
    }
    assert!(results.len() <= 20);
}

// ── step 4: hit the live HTTP API ─────────────────────────────────────────────

#[tokio::test]
async fn test_e2e_api_health() {
    let client = reqwest::Client::new();
    let res = client.get("http://localhost:3000/health").send().await;

    match res {
        Ok(r) => {
            assert_eq!(r.status(), 200);
            let body: Value = r.json().await.unwrap();
            assert_eq!(body["status"], "ok");
            println!("Health check passed");
        }
        Err(_) => println!("Server not running — skipping. Start with `cargo run`"),
    }
}

#[tokio::test]
async fn test_e2e_api_text_search() {
    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:3000/search/text?limit=20")
        .json(&serde_json::json!({ "query": "casual summer dress" }))
        .send()
        .await;

    match res {
        Ok(r) => {
            assert_eq!(r.status(), 200, "Expected 200, got {}", r.status());
            let body: Value = r.json().await.unwrap();
            let results = body["results"].as_array().unwrap();
            println!("\nAPI text search results for 'casual summer dress':");
            for r in results {
                println!("  score={} | {}", r["score"], r["description"]);
            }
            assert!(!results.is_empty(), "API returned no results");
        }
        Err(_) => println!("Server not running — skipping. Start with `cargo run`"),
    }
}

#[tokio::test]
async fn test_e2e_api_image_search() {
    let client = reqwest::Client::new();
    let image_bytes = fs::read(SAMPLE_IMAGE).expect("sample.jpg not found");

    let form = reqwest::multipart::Form::new()
        .part("image", reqwest::multipart::Part::bytes(image_bytes).file_name("sample.jpg"));

    let res = client
        .post("http://localhost:3000/search/image?limit=20")
        .multipart(form)
        .send()
        .await;

    match res {
        Ok(r) => {
            assert_eq!(r.status(), 200, "Expected 200, got {}", r.status());
            let body: Value = r.json().await.unwrap();
            let results = body["results"].as_array().unwrap();
            println!("\nAPI image search results:");
            for r in results {
                println!("  score={} | {}", r["score"], r["description"]);
            }
        }
        Err(_) => println!("Server not running — skipping. Start with `cargo run`"),
    }
}
