"""Tests for Phase 1: Preprocessing."""

import pytest
import cv2
import numpy as np
from src.preprocessing import Preprocessor
from src.utils.image_io import load_image

CONFIG = {
    'bilateral_d': 9,
    'bilateral_sigma_color': 75,
    'bilateral_sigma_space': 75,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': [8, 8],
    'max_working_dim': 1024
}

@pytest.fixture
def preprocessor():
    return Preprocessor(config=CONFIG, debug=False)

@pytest.fixture
def sample_bgr():
    """Synthetic 200×200 BGR image: white circle on black background."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 60, (200, 200, 200), -1)
    return img

# TC-1.1: Grayscale output is single-channel
def test_grayscale_output_shape(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    assert gray.ndim == 2, "Grayscale must be 2D (H×W)"
    assert gray.shape == (200, 200)

# TC-1.2: CLAHE output has same shape and dtype as input
def test_clahe_preserves_shape(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    clahe = preprocessor.apply_clahe(gray)
    assert clahe.shape == gray.shape
    assert clahe.dtype == np.uint8

# TC-1.3: Bilateral filter produces non-identical but same-shape output
def test_bilateral_filter_smooths_image(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    blurred = preprocessor.bilateral_filter(gray)
    assert blurred.shape == gray.shape
    assert not np.array_equal(blurred, gray), "Blur should change pixel values"

# TC-1.4: Sobel magnitude output is binary (0 or 255 only)
def test_sobel_is_binary(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    blurred = preprocessor.bilateral_filter(gray)
    edges = preprocessor.sobel_magnitude(blurred)
    unique_vals = np.unique(edges)
    assert set(unique_vals).issubset({0, 255}), f"Sobel has non-binary values: {unique_vals}"

# TC-1.5: Sobel magnitude detects edges on synthetic circle image
def test_sobel_detects_circle_edges(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    blurred = preprocessor.bilateral_filter(gray)
    edges = preprocessor.sobel_magnitude(blurred)
    # Should have significant white pixels (circle boundary detected)
    white_pixel_ratio = np.sum(edges > 0) / edges.size
    assert white_pixel_ratio > 0.01, "Sobel found no edges on synthetic circle"
    assert white_pixel_ratio < 0.30, "Too many edges — noise or wrong thresholds"

# TC-1.6: Scale normalization respects max_dim
def test_scale_normalization(preprocessor):
    large_img = np.zeros((2000, 3000, 3), dtype=np.uint8)
    scaled = preprocessor.normalize_scale(large_img)
    assert max(scaled.shape[:2]) <= 1024, "Scale should limit max dimension to 1024"

# TC-1.7: Small images are not upscaled
def test_small_image_not_upscaled(preprocessor):
    small_img = np.zeros((100, 100, 3), dtype=np.uint8)
    scaled = preprocessor.normalize_scale(small_img)
    assert scaled.shape == small_img.shape, "Small images should not be resized"

# TC-1.8: FileNotFoundError on missing file
def test_missing_file_raises(preprocessor):
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/image.jpg")

# TC-1.9: run() returns all required keys
def test_run_returns_all_keys(preprocessor, tmp_path):
    # Save synthetic image to temp path
    img_path = str(tmp_path / "test.jpg")
    sample = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(sample, (100, 100), 60, (180, 180, 180), -1)
    cv2.imwrite(img_path, sample)
    
    result = preprocessor.run(img_path)
    required_keys = {'original', 'scaled', 'gray', 'clahe', 'blurred', 'edges', 'scale_factor'}
    assert required_keys.issubset(result.keys())

# TC-1.10: Scale factor is correct value
def test_scale_factor_accuracy(preprocessor, tmp_path):
    # 2048-wide image → should scale to 1024 (factor = 0.5)
    img_path = str(tmp_path / "large.jpg")
    large = np.zeros((1080, 2048, 3), dtype=np.uint8)
    cv2.imwrite(img_path, large)
    result = preprocessor.run(img_path)
    assert abs(result['scale_factor'] - 0.5) < 0.01
