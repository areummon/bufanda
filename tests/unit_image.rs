/// Tests for image preprocessing
/// Requirements: a sample jpg at tests/fixtures/sample.jpg
/// Run with: cargo test --test unit_image

use fashion_search::processor::image::preprocess_image;

#[test]
fn test_preprocess_output_shape() {
    let tensor = preprocess_image("tests/fixtures/sample.jpg");
    // CLIP expects (batch=1, channels=3, height=224, width=224)
    assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
}

#[test]
fn test_preprocess_values_are_normalized() {
    let tensor = preprocess_image("tests/fixtures/sample.jpg");
    // After CLIP normalization values should be roughly in [-3.0, 3.0]
    // (mean ~0.45, std ~0.27, so raw [0,1] maps to roughly [-1.7, 2.0])
    for val in tensor.iter() {
        assert!(
            *val > -4.0 && *val < 4.0,
            "Pixel value {} is outside expected normalized range",
            val
        );
    }
}

#[test]
fn test_preprocess_no_nan_or_inf() {
    let tensor = preprocess_image("tests/fixtures/sample.jpg");
    for val in tensor.iter() {
        assert!(!val.is_nan(), "Tensor contains NaN");
        assert!(!val.is_infinite(), "Tensor contains Inf");
    }
}

#[test]
fn test_preprocess_three_channels_differ() {
    let tensor = preprocess_image("tests/fixtures/sample.jpg");

    let mut ch0 = Vec::new();
    let mut ch1 = Vec::new();

    for y in 0..224 {
        for x in 0..224 {
            ch0.push(tensor[[0, 0, y, x]]);
            ch1.push(tensor[[0, 1, y, x]]);
        }
    }

    assert_ne!(ch0, ch1, "Channels 0 and 1 are identical — image may be greyscale or preprocessing is wrong");
}
