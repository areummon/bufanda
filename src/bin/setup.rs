//! One time setup binary that creates the Qdrant Collection with dual named vector
//! spaces for image and text embeddings.
//!
//! Usage: 
//!     cargo run --bin setup                   # creates "outfits" collection
//!     cargo run --bin setup -- outfits_tes    # creates "outfits_test" collection

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, VectorParamsBuilder, VectorsConfigBuilder 
};
use dotenvy::dotenv;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    let url = std::env::var("QDRANT_URL")?;
    let api_key = std::env::var("QDRANT_API_KEY")?;

    // Collection name can be passed as a cli argument, for example:
    // cargo run --bin setup -- outfits_test
    // Defaults to "outfits" (because thats what I used) if no argument is given.
    let collection = std::env::args().nth(1).unwrap_or("outfits".to_string());

    let client = Qdrant::from_url(&url)
        .api_key(api_key)
        .build()?;

    if client.collection_exists(&collection).await? {
        println!("Collection '{}' already exists, skipping.", collection);
        return Ok(());
    }

    // Two named vector spaces - one for CLIP image embeddings, one for CLIP text
    // embeddings of VLM captions. Both 512-dimensional with cosine similarity,
    // matching FashionCLIP's output.
    let mut vectors_config = VectorsConfigBuilder::default();
    vectors_config.add_named_vector_params(
        "image", 
        VectorParamsBuilder::new(512, Distance::Cosine).build(),
    );
    vectors_config.add_named_vector_params(
        "text", 
        VectorParamsBuilder::new(512, Distance::Cosine).build(),
    );

    client
        .create_collection(
            CreateCollectionBuilder::new(&collection)
                .vectors_config(vectors_config)    
        )
        .await?;

        println!("Collection {} created.", collection);
        Ok(())
}
