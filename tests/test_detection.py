"""Tests for Phase 2: Detection."""

import pytest
import cv2
import numpy as np
from src.detection import CoinDetector
from src.utils.geometry import circle_iou

CONFIG = {
    'dp': 1.2,
    'min_dist': 30,
    'param1': 80,
    'param2': 30,
    'min_radius': 15,
    'max_radius': 150,
    'nms_iou_threshold': 0.3,
}

@pytest.fixture
def detector():
    return CoinDetector(config=CONFIG, debug=False)

def make_circle_image(centers_radii: list, size=(400, 400)) -> np.ndarray:
    """Create synthetic grayscale image with filled white circles."""
    img = np.zeros(size, dtype=np.uint8)
    for (cx, cy, r) in centers_radii:
        cv2.circle(img, (cx, cy), r, 255, -1)
    return cv2.GaussianBlur(img, (5, 5), 1.5)

# TC-2.1: Single circle detection
def test_single_circle_detected(detector):
    img = make_circle_image([(200, 200, 50)])
    raw = detector.detect_circles(img)
    assert raw is not None, "Should detect at least one circle"
    assert raw.shape[1] >= 1

# TC-2.2: Multiple non-overlapping circles
def test_multiple_circles_detected(detector):
    img = make_circle_image([(100, 100, 40), (300, 100, 40), (200, 300, 40)])
    raw = detector.detect_circles(img)
    assert raw is not None
    circles = [(int(x), int(y), int(r)) for x, y, r in raw[0]]
    from src.utils.geometry import nms_circles
    nms_result = nms_circles(circles)
    # Should find approximately 3 circles
    assert 2 <= len(nms_result) <= 4, f"Expected ~3 circles, got {len(nms_result)}"

# TC-2.3: NMS removes overlapping duplicates
def test_nms_removes_duplicates(detector):
    from src.utils.geometry import nms_circles
    # Two nearly identical circles (same center, slightly diff radius)
    circles = [(100, 100, 40), (102, 101, 42), (300, 300, 40)]
    kept = nms_circles(circles, threshold=0.3)
    assert len(kept) == 2, f"NMS should keep 2, got {len(kept)}"

# TC-2.4: Circle overlap function — no overlap
def test_circle_overlap_none():
    c1, c2 = (0, 0, 30), (100, 0, 30)
    overlap = circle_iou(c1, c2)
    assert overlap == 0.0

# TC-2.5: Circle overlap function — identical circles
def test_circle_overlap_identical():
    c = (50, 50, 30)
    overlap = circle_iou(c, c)
    assert abs(overlap - 1.0) < 0.05, f"Identical circles should have ~1.0 overlap, got {overlap}"

# TC-2.6: Boundary filtering removes edge circles
def test_boundary_filter(detector):
    # Circle at (5, 5, 40) — center near top-left, mostly outside
    circles = [(5, 5, 40), (200, 200, 40)]
    filtered = detector.filter_boundary_circles(circles, image_shape=(400, 400))
    # The (5,5) circle should be removed
    assert (200, 200, 40) in filtered
    assert not any(x < 10 and y < 10 for x, y, r in filtered)

# TC-2.7: Empty image returns empty list (raw might be None)
def test_no_circles_in_blank_image(detector):
    blank = np.zeros((400, 400), dtype=np.uint8)
    raw = detector.detect_circles(blank)
    assert raw is None or (raw is not None and len(raw[0]) == 0)

# TC-2.8: Radius range suggestion is reasonable
def test_radius_range_suggestion():
    min_r, max_r = CoinDetector.suggest_radius_range((1080, 1920))
    diag = np.sqrt(1080**2 + 1920**2)
    assert min_r >= int(diag * 0.025)
    assert max_r <= int(diag * 0.26)
    assert min_r < max_r

# TC-2.9: Visualization returns BGR image with correct shape
def test_visualization_shape(detector):
    original = np.zeros((400, 400, 3), dtype=np.uint8)
    circles = [(100, 100, 40), (250, 250, 35)]
    viz = detector.visualize_detections(original, circles)
    assert viz.shape == original.shape
    assert viz.dtype == np.uint8
    # Visualization should have modified pixels (green circles drawn)
    assert not np.array_equal(viz, original)

# TC-2.10: Detected radius within 15% of true radius
def test_detection_radius_accuracy(detector):
    true_r = 50
    img = make_circle_image([(200, 200, true_r)])
    raw = detector.detect_circles(img)
    assert raw is not None
    detected_r = raw[0][0][2]
    assert abs(detected_r - true_r) / true_r < 0.15, \
        f"Detected radius {detected_r} too far from true {true_r}"
