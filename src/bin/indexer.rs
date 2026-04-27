//! Indexes DeepFashion2 validation images into Qdrant for semanti search.
//!
//! For each "user" photo it: generates a caption using llava:7b, encodes the iamge with
//! FashionCLIP's image encode, encodes the caption with FashionCLIP's text encode and lastly,
//! upserts both vectors into Qdrant with metadata.
//!
//! Resumes automatically from `indexer_progress.log` if interrupted.
//! Only indexes `source: "user"` images - "shop" photos are skipped.
//!
//! Usage: cargo run --bin indexer

use fashion_search::captioner::Captioner;
use fashion_search::processor::model::BufandaAI;
use fashion_search::processor::image::preprocess_image;
use fashion_search::db::VectorDB;
use std::fs;
use std::io::Write;
use dotenvy::dotenv;
use serde_json::Value;
use rand::seq::SliceRandom;
use rand::rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let mut ai = BufandaAI::new(
        "models/text/model.onnx",
        "models/image/model.onnx",
        "models/text/tokenizer.json",
    );
    let db = VectorDB::new(
        &std::env::var("QDRANT_URL")?,
        &std::env::var("QDRANT_API_KEY")?,
        &std::env::var("COLLECTION").unwrap_or("outfits".to_string()),
    );
    let captioner = Captioner::new();

    // ── resume: load already processed file stems ─────────────────────────
    let progress_file = "./indexer_progress.log";
    let processed: std::collections::HashSet<String> = fs::read_to_string(progress_file)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();

    println!("▶ Resuming — {} images already indexed, skipping those.", processed.len());
    // ─────────────────────────────────────────────────────────────────────

    let image_dir = "./data/validation/image";
    let json_dir  = "./data/validation/annos";

    let mut entries: Vec<_> = fs::read_dir(image_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_file()
                && e.path().extension().map_or(false, |ext| ext == "jpg")
        })
        .collect();

    // DeepFashion2 groups consecutive images by garment, so I used a Shuffle for result diversity.
    entries.shuffle(&mut rng());

    let total = entries.len();
    // Start the indexed counter from alerady processed images.
    let mut indexed = processed.len();

    for entry in entries {
        let img_path  = entry.path();
        let file_stem = img_path.file_stem().unwrap().to_str().unwrap().to_string();
        let json_path = format!("{}/{}.json", json_dir, file_stem);
        let path_str  = img_path.to_str().unwrap();

        // skip already indexed — prevents duplicates on resume
        if processed.contains(&file_stem) {
            continue;
        }

        // skip shop images
        let json_content = match fs::read_to_string(&json_path) {
            Ok(s) => s,
            Err(_) => {
                println!("No annotation for {}, skipping.", file_stem);
                continue;
            }
        };

        let parsed: Value = match serde_json::from_str(&json_content) {
            Ok(v) => v,
            Err(_) => {
                println!("Bad JSON for {}, skipping.", file_stem);
                continue;
            }
        };

        let source = parsed
            .get("source")
            .and_then(|s| s.as_str())
            .unwrap_or("");

        if source != "user" {
            continue;
        }

        // category label from annotations (kept as metadata)
        let category = parsed
            .as_object()
            .map(|obj| {
                obj.iter()
                    .filter(|(k, _)| k.starts_with("item"))
                    .filter_map(|(_, val)| {
                        val.get("category_name")
                            .and_then(|c| c.as_str())
                            .map(String::from)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        // VLM caption with fallback to category label
        let description = captioner
            .caption(path_str)
            .await
            .unwrap_or_else(|_| category.clone());

        let image_vector = ai.encode_image(preprocess_image(path_str));
        let text_vector  = ai.encode_text(&description);

        // retry loop — recovers from transient network errors
        let mut attempts = 0;
        let upserted = loop {
            match db.upsert_outfit(
                image_vector.clone(),
                text_vector.clone(),
                path_str,
                &description,
                &category,
            ).await {
                Ok(_) => break true,
                Err(e) if attempts < 3 => {
                    attempts += 1;
                    println!(
                        "Upsert failed (attempt {}/3): {} — retrying in 5s",
                        attempts, e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
                Err(e) => {
                    println!("Skipping {} after 3 failed attempts: {}", file_stem, e);
                    break false;
                }
            }
        };

        if upserted {
            // append to progress file so next run skips this image
            let mut file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(progress_file)?;
            writeln!(file, "{}", file_stem)?;

            indexed += 1;
            println!("[{}/{}] {} → {}", indexed, total, file_stem, description);
        }
    }

    println!("All done. {} images indexed.", indexed);
    Ok(())
}
