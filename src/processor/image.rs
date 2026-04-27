//! Preprocessing pipeline for images fed into the CLIP image encoder.

use ndarray::Array4;

/// The preprocessing pipeline is as follows: opens the image from the given path,
/// resizes it to 224x224 pixels using triangle (bilinear) filtering, 
/// converts it to RGB, scales pixel values from `[0, 255]` to `[0.0, 1.0]`, and
/// then normalizes each channel using CLIP's mean and standard deviation.
///
/// # Arguments
/// * 'path' - Path to the image file (`.jpg`, `.png`, etc.)
///
/// # Returns
///
/// A 4-dimensional tensor of shape `(1, 3, 224, 224)` in NHCW format
/// (batch, channels, height, width) ready to be fed into the CLIP image encoder.
///
/// # Panics
///
/// Panics if the file does not existe or cannot be decoded as an image.
pub fn preprocess_image(path: &str) -> Array4<f32> {
    let img = image::open(path).expect("Failed to open image");
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
    let rgb_img = resized.to_rgb8();

    // values per channel: R, G, B
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];

    let mut tensor = Array4::zeros((1, 3, 224, 224));
    for (x, y, pixel) in rgb_img.enumerate_pixels() {
        for c in 0..3 {
            let channel_val = pixel[c] as f32 / 255.0;
            // Standard normalization
            tensor[[0, c, y as usize, x as usize]] = (channel_val - mean[c]) / std[c]
        }
    }

    tensor
}
