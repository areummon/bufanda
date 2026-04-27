//! Recovery utility that reconstructs `indexer_progress.log` from the already-indexed points in
//! Qdrant.
//!
//! Run this if the progress file is lost or out of sync with the database to avoid
//! re-indexing images that are already in the collection.
//!
//! Usage: cargo run --bin rebuild_progress

use dotenvy::dotenv;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let client = qdrant_client::Qdrant::from_url(&std::env::var("QDRANT_URL")?)
        .api_key(std::env::var("QDRANT_API_KEY")?)
        .build()?;

    let mut offset: Option<qdrant_client::qdrant::PointId> = None;
    let mut total = 0;
    let mut file = std::fs::File::create("./indexer_progress.log")?;

    // Qdrant doen't support fetching all points at once, so we scroll
    // through them in pages of 100, using the last point's ID as the offset for the next page.
    loop {
        let mut builder = qdrant_client::qdrant::ScrollPointsBuilder::new("outfits")
            .with_payload(true)
            .limit(100);

        if let Some(ref o) = offset {
            builder = builder.offset(o.clone());
        }

        let result = client.scroll(builder).await?;

        for point in &result.result {
            if let Some(url) = point.payload.get("image_url")
                .and_then(|v| v.kind.as_ref())
                .and_then(|k| {
                    if let qdrant_client::qdrant::value::Kind::StringValue(s) = k {
                        Some(s)
                    } else {
                        None
                    }
                })
            {
                // Extract jus the file stem ("002115" from your image folder) since that is the
                // format the indexer uses to track progress.
                if let Some(stem) = std::path::Path::new(url.as_str())
                    .file_stem()
                    .and_then(|s| s.to_str())
                {
                    writeln!(file, "{}", stem)?;
                    total += 1;
                }
            }
        }

        // next_page_offset is None when it reaches the last page.
        match result.next_page_offset {
            Some(next) => offset = Some(next),
            None => break,
        }
    }

    println!("Done. Rebuilt progress file with {} entries.", total);
    Ok(())
}
