use fashion_search::processor::model::BufandaAI;
use fashion_search::db::VectorDB;
use fashion_search::engine::SearchEngine;
use fashion_search::api::{router, AppState};
use dotenvy::dotenv;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() {
    // Necessary to work with connections.
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");
    // Checks that you have a .env defined in your project directory.
    dotenv().ok();

    // You should define the url and the api_key in your .env for these to work.
    let qdrant_url = std::env::var("QDRANT_URL")
        .expect("QDRANT_URL not set in .env");
    let qdrant_key = std::env::var("QDRANT_API_KEY")
        .expect("QDRANT_API_KEY not set in .env");
    let port = std::env::var("PORT").unwrap_or("3000".to_string());

    println!("Loading CLIP models...");
    let ai = BufandaAI::new(
        "models/text/model.onnx",
        "models/image/model.onnx",
        "models/text/tokenizer.json",
    );
    println!("Models loaded.");

    // In my case, I chose to define a COLLECTION variable in my .env so I could
    // change it in case I need to use another collection in my cluster (vector db).
    let collection = std::env::var("COLLECTION").unwrap_or("outfits".to_string());
    let db = VectorDB::new(&qdrant_url, &qdrant_key, &collection);
    let engine = SearchEngine::new(ai, db);
    // As explained in the api module, this is Arc<Mutex<>> so it can be safely shared across
    // concurrent HTTP request. ONNX sessions are not thread-safe.
    let state: AppState = Arc::new(Mutex::new(engine));

    let app = router(state);
    let addr = format!("0.0.0.0:{}",port);

    println!("Server running on http://{}", addr);
    println!("Endpoints:");
    println!("  GET  /health");
    println!("  POST /search/text   {{ \"query\": \"cute summer dress\" }}");
    println!("  POST /search/image  multipart/form-data with image field");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("Failed to bind port");

    axum::serve(listener, app)
        .await
        .expect("Server error");
}
