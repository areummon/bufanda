//! CLIP-based encoder wrapping two ONNX sessions (text and image)
//! that produce normalized 512-dimensional embeddings for semantic search.

use ort::session::Session;
use ort::value::TensorRef; 
use ort::session::builder::GraphOptimizationLevel;
use ndarray::{Array2, Array4};
use tokenizers::Tokenizer;

/// Wrapper around two ONNX inference sessions and a tokenizer
/// for the FashionCLIP model.
///
/// Encodes both text queries and images into a shared 512-dimensional
/// embedding space, enabling semantic similarity search between them.
pub struct BufandaAI {
    text_session: Session,
    image_session: Session,
    tokenizer: Tokenizer,
}

impl BufandaAI {
    /// The constructor for the AI model.
    ///
    /// # Arguments
    ///
    /// * `text_path` - Path to the text model.
    /// * `image_path` - Path to the image model.
    /// * `tokenizer_path` - Path to the tokenizer
    ///
    /// # Returns
    ///
    /// A new instance of BufandaAI or the AI model.
    ///
    /// # Panics
    ///
    /// Panics if any of the model files or the tokenizer file do not exist
    /// or cannot be loaded as valid ONNX/tokenizer files.
    ///
    pub fn new(text_path: &str, image_path: &str, tokenizer_path: &str) -> Self {
        let text_session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(text_path)
            .unwrap();

        let image_session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .commit_from_file(image_path)
            .unwrap();

        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        Self { text_session, image_session, tokenizer }
    }

    /// Encodes a text using the model and the format of the AI model chosen.
    ///
    /// In my case I used a CLIP model fine-tuned for fashion named FashionCLIP
    /// And it need three arguments to be passed as input: the input tokens, 
    /// the pixel values, and a mask head attention array.
    ///
    /// In the case of encoding text, the pixel values don't need a value. So it
    /// creates an array of the corresponding dimension with only zeroes.
    ///
    /// After that, it gets the outputs of the encoding, and returns a normalized
    /// version of the values.
    ///
    /// # Arguments
    ///
    /// * `text` - A string of natural language text with a max of ~77 tokens due to CLIP's context
    /// window
    ///
    /// # Returns 
    /// 
    /// A Vector of normalized float values that represents the encoding.
    ///
    pub fn encode_text(&mut self, text: &str) -> Vec<f32> {
        let encoding = self.tokenizer.encode(text, true).unwrap();
        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        
        let seq_len = input_ids.len();
        let ids_array = Array2::from_shape_vec((1, seq_len), input_ids).unwrap();
        let mask_array = Array2::from_shape_vec((1, seq_len), attention_mask).unwrap();
        // Needed for the model
        let pixel_values = Array4::<f32>::zeros((1, 3, 224, 224));

        let outputs = self.text_session
            .run(ort::inputs![
                "input_ids" => TensorRef::from_array_view(&ids_array).unwrap(),
                "pixel_values" => TensorRef::from_array_view(&pixel_values).unwrap(),
                "attention_mask" => TensorRef::from_array_view(&mask_array).unwrap()
            ])
            .unwrap();

        let view = outputs["text_embeds"].try_extract_array::<f32>().unwrap();
        normalize(view.iter().cloned().collect())
    }

    /// Encodes an image using the model and the format of the AI model chosed.
    ///
    /// In my case I used a CLIP model fine-tuned for fashio, named FashionCLIP.
    /// And it need three arguments to be passed as input: the input tokens, 
    /// the pixel values, and a mask head attention array.
    ///
    /// In the case of encoding image, the only values we have are the pixel values
    /// so we need to create arrays for the tokenized text values and the mask attention value with
    /// the corresponding dimensions, they dont need any special values, so it creates it with
    /// zeros.
    ///
    /// After that, it gets the outputs of the encoding, and returns a normalized
    /// version of the values.
    ///
    /// # Arguments
    ///
    /// * `image_tensor` - A preprocessed image tensor of shape `(1, 3, 224, 224)` as produced by
    /// [`preprocess_image`](crate::processor::image::preprocess_image)
    ///
    /// # Returns 
    /// 
    /// A Vector of normalized float values that represents the encoding.
    ///
    pub fn encode_image(&mut self, image_tensor: Array4<f32>) -> Vec<f32> {
        // Values needed for the model, only filled with zeroes.
        let input_ids = Array2::<i64>::zeros((1, 77));
        let mask_array = Array2::<i64>::zeros((1, 77));

        let outputs = self.image_session
            .run(ort::inputs![
                "input_ids" => TensorRef::from_array_view(&input_ids).unwrap(),
                "pixel_values" => TensorRef::from_array_view(&image_tensor).unwrap(),
                "attention_mask" => TensorRef::from_array_view(&mask_array).unwrap()
            ])
            .unwrap();

        let view = outputs["image_embeds"].try_extract_array::<f32>().unwrap();
        normalize(view.iter().cloned().collect())
    }
}

/// Function to L2-normalize a vector.
///
/// Ensures cosine similarity can be computed as a simple dot product between two vectors.
fn normalize(vec: Vec<f32>) -> Vec<f32> {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    vec.into_iter().map(|x| x / norm).collect()
}
