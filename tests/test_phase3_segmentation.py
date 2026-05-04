# tests/test_phase3_segmentation.py
import pytest
import cv2
import numpy as np
from src.phase3_segmentation import CoinSegmenter

CONFIG = {
    'target_size': 128,
    'padding_ratio': 0.05,
    'mask_erosion_px': 2,
}

@pytest.fixture
def segmenter():
    return CoinSegmenter(config=CONFIG, debug=False)

@pytest.fixture
def sample_image():
    """400×400 BGR image with one gold-colored coin (circle) at center."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    # Draw gold-ish circle
    cv2.circle(img, (200, 200), 80, (30, 165, 210), -1)  # BGR gold
    return img

# TC-3.1: ROI crop returns correct size
def test_crop_roi_shape(segmenter, sample_image):
    roi, offset = segmenter.crop_roi(sample_image, 200, 200, 80)
    # Should be approximately 2*(80 + 4) × 2*(80 + 4) = 168×168
    expected_size = 2 * (80 + int(80 * 0.05)) + 1
    assert abs(roi.shape[0] - expected_size) <= 2
    assert abs(roi.shape[1] - expected_size) <= 2

# TC-3.2: Circular mask has correct shape and dtype
def test_circular_mask_properties(segmenter, sample_image):
    roi, (x1, y1) = segmenter.crop_roi(sample_image, 200, 200, 80)
    cx_local = 200 - x1
    cy_local = 200 - y1
    _, mask = segmenter.apply_circular_mask(roi, cx_local, cy_local, 80)
    assert mask.ndim == 2
    assert mask.dtype == np.uint8
    unique = np.unique(mask)
    assert set(unique).issubset({0, 255})

# TC-3.3: Mask covers correct area (approximately πr²)
def test_mask_area(segmenter, sample_image):
    roi, (x1, y1) = segmenter.crop_roi(sample_image, 200, 200, 80)
    cx_local = 200 - x1
    cy_local = 200 - y1
    _, mask = segmenter.apply_circular_mask(roi, cx_local, cy_local, 78)  # 80 - erosion=2
    white_pixels = np.sum(mask > 0)
    expected_area = np.pi * 78**2
    # Allow 10% tolerance
    assert abs(white_pixels - expected_area) / expected_area < 0.10

# TC-3.4: Masked ROI has zero background pixels in corners
def test_masked_roi_corners_black(segmenter, sample_image):
    roi, (x1, y1) = segmenter.crop_roi(sample_image, 200, 200, 80)
    cx_local, cy_local = 200 - x1, 200 - y1
    masked_roi, _ = segmenter.apply_circular_mask(roi, cx_local, cy_local, 80)
    # Top-left corner should be black (outside circle)
    corner = masked_roi[0:5, 0:5]
    assert np.all(corner == 0), "Corners outside circle should be black"

# TC-3.5: Normalized ROI is exactly target_size × target_size
def test_normalize_shape(segmenter, sample_image):
    roi, (x1, y1) = segmenter.crop_roi(sample_image, 200, 200, 80)
    cx_local, cy_local = 200 - x1, 200 - y1
    masked_roi, mask = segmenter.apply_circular_mask(roi, cx_local, cy_local, 80)
    norm_roi, norm_mask = segmenter.normalize_roi(masked_roi, mask)
    assert norm_roi.shape == (128, 128, 3)
    assert norm_mask.shape == (128, 128)

# TC-3.6: Normalized mask is still binary after resize
def test_norm_mask_binary(segmenter, sample_image):
    roi, (x1, y1) = segmenter.crop_roi(sample_image, 200, 200, 80)
    cx_local, cy_local = 200 - x1, 200 - y1
    masked_roi, mask = segmenter.apply_circular_mask(roi, cx_local, cy_local, 80)
    _, norm_mask = segmenter.normalize_roi(masked_roi, mask)
    unique = np.unique(norm_mask)
    assert set(unique).issubset({0, 255})

# TC-3.7: segment_coin returns all required keys
def test_segment_coin_keys(segmenter, sample_image):
    seg = segmenter.segment_coin(sample_image, (200, 200, 80), coin_index=0)
    required = {'index', 'circle', 'roi', 'masked_roi', 'mask', 'norm_roi', 'norm_mask', 'local_center'}
    assert required.issubset(seg.keys())

# TC-3.8: Clamping prevents out-of-bounds crops for edge coins
def test_boundary_coin_no_crash(segmenter, sample_image):
    # Coin at corner, partially outside image
    roi, offset = segmenter.crop_roi(sample_image, 5, 5, 80)
    assert roi.size > 0, "Boundary coin crop should be non-empty"
    assert roi.shape[0] > 0 and roi.shape[1] > 0

# TC-3.9: Batch run returns correct count
def test_run_returns_correct_count(segmenter, sample_image):
    # Mock phase1 and phase2 results
    p1 = {'scaled': sample_image}
    p2 = {'circles': [(100, 100, 40), (300, 300, 40)]}
    result = segmenter.run(p1, p2)
    assert result['count'] == 2
    assert len(result['segments']) == 2

# TC-3.10: Grid image has correct dimensions
def test_grid_image_dimensions(segmenter):
    t = 128
    # 3 coins → 1 row of 8 columns → 1 row × 8 cols = 128 × 1024
    fake_seg = [{'norm_roi': np.zeros((t, t, 3), dtype=np.uint8)} for _ in range(3)]
    grid = segmenter._make_coin_grid(fake_seg, cols=8)
    assert grid.shape == (t * 1, t * 8, 3), f"Unexpected grid shape: {grid.shape}"