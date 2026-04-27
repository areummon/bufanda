//! A module connecting the database and the AI model to make queries easily.

use crate::processor::model::BufandaAI;
use crate::processor::image::preprocess_image;
use crate::db::{VectorDB, SearchResult};

pub struct SearchEngine {
    pub ai: BufandaAI,
    pub db: VectorDB,
}

impl SearchEngine {
    pub fn new(ai: BufandaAI, db: VectorDB) -> Self {
        Self { ai, db }
    }

    /// # Errors
    ///
    /// Returns an error if the Qdrant search request fails.
    pub async fn search_by_text(&mut self, query: &str, limit: u64) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let vector = self.ai.encode_text(query);
        self.db.search_by_text_vector(vector, limit).await
    }

    /// # Errors
    ///
    /// Returns an error if the Qdrant search request fails.
    pub async fn search_by_image(&mut self, image_path: &str, limit: u64) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let tensor = preprocess_image(image_path);
        let vector = self.ai.encode_image(tensor);
        self.db.search_by_image_vector(vector, limit).await
    }
}
