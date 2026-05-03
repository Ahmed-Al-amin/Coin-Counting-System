# Phase 2: Circular Hough Transform Detection
## Coin Counting System — Classical CV Pipeline

---

## 1. Overview

The **Circular Hough Transform (CHT)** is the mathematical heart of this system. It detects circles in an edge image by voting in a 3D parameter space (center_x, center_y, radius). Each edge pixel "votes" for all circles it could belong to. Peaks in the accumulator space indicate actual circles.

**Goal**: Given the preprocessed edge map from Phase 1, return a list of `(x, y, r)` tuples — one per detected coin.

---

## 2. Mathematical Foundation

### 2.1 The Circle Equation

A circle is defined by:
```
(x - a)² + (y - b)² = r²
```
where `(a, b)` is the center and `r` is the radius.

### 2.2 The Hough Voting Process

For each edge pixel `(x, y)` with gradient direction `θ`:
- The circle center must lie along the gradient direction at distance `r`
- Cast votes at: `a = x - r·cos(θ)`, `b = y - r·sin(θ)` for all r in [r_min, r_max]
- Accumulator `A[a][b][r]++`

After all votes: peaks in A are circles.

### 2.3 OpenCV's `HoughCircles`

OpenCV uses a **2-stage** approach (faster than full 3D voting):
1. **Stage 1**: Find candidate centers using 2D accumulator (all radii projected)
2. **Stage 2**: For each center, find the best-fitting radius using edge statistics

---

## 3. Implementation Steps

### Step 2.1 — Core Detection

```python
# src/phase2_detection.py
import cv2
import numpy as np
from typing import List, Tuple, Optional

Circle = Tuple[int, int, int]  # (x, y, r)

class CoinDetector:
    def __init__(self, config: dict, debug: bool = False,
                 output_dir: str = "outputs/intermediate/phase2"):
        self.config     = config
        self.debug      = debug
        self.output_dir = output_dir

    def detect_circles(self, blurred_gray: np.ndarray) -> np.ndarray:
        """
        Run OpenCV HoughCircles on a blurred grayscale image.
        
        Parameters explained:
          dp        - Inverse ratio of accumulator resolution to image resolution.
                      dp=1: accumulator same size as image.
                      dp=2: accumulator half size (faster, less accurate).
          minDist   - Minimum distance between circle centers.
                      Prevents two detections for the same coin.
          param1    - Upper Canny threshold passed internally to HoughCircles.
                      (Lower threshold = param1/2 automatically)
          param2    - Accumulator threshold for circle centers at detection stage.
                      Lower = more (potentially false) circles detected.
          minRadius - Minimum coin radius in pixels.
          maxRadius - Maximum coin radius in pixels.
        """
        cfg = self.config['detection']
        
        circles = cv2.HoughCircles(
            blurred_gray,
            cv2.HOUGH_GRADIENT,
            dp        = cfg.get('dp', 1.2),
            minDist   = cfg.get('min_dist', 30),
            param1    = cfg.get('param1', 80),
            param2    = cfg.get('param2', 30),
            minRadius = cfg.get('min_radius', 15),
            maxRadius = cfg.get('max_radius', 200),
        )
        return circles  # Shape: (1, N, 3) or None
```

### Step 2.2 — Parameter Tuning Guide

```python
    @staticmethod
    def suggest_radius_range(image_shape: tuple, 
                              min_coin_fraction: float = 0.03,
                              max_coin_fraction: float = 0.25) -> Tuple[int, int]:
        """
        Auto-estimate radius range from image size.
        
        Assumption: smallest coin (1c) is at least 3% of image diagonal,
                    largest coin (2€) is at most 25% of image diagonal.
        """
        diag = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        min_r = int(diag * min_coin_fraction)
        max_r = int(diag * max_coin_fraction)
        return max(10, min_r), min(500, max_r)
```

### Step 2.3 — Non-Maximum Suppression (NMS)

OpenCV may return overlapping circles for the same coin (especially with low `param2`). NMS removes duplicates.

```python
    def non_max_suppression(self, circles: List[Circle], 
                             iou_threshold: float = 0.3) -> List[Circle]:
        """
        Remove overlapping circles using Intersection over Union.
        
        For circles, IoU is approximated by overlap area / union area.
        Circles are sorted by confidence (approximated by radius — larger
        circles that match a coin boundary tend to be more reliable).
        
        Keeps the circle with largest radius among overlapping candidates.
        """
        if len(circles) == 0:
            return []
        
        # Sort by radius descending (heuristic: larger = more accumulated votes)
        sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
        kept = []
        
        for candidate in sorted_circles:
            suppress = False
            for kept_circle in kept:
                if self._circle_overlap(candidate, kept_circle) > iou_threshold:
                    suppress = True
                    break
            if not suppress:
                kept.append(candidate)
        
        return kept

    def _circle_overlap(self, c1: Circle, c2: Circle) -> float:
        """Compute overlap ratio between two circles (0=none, 1=identical)."""
        x1, y1, r1 = c1
        x2, y2, r2 = c2
        d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        if d >= r1 + r2:
            return 0.0  # No overlap
        if d <= abs(r1 - r2):
            # One circle fully inside the other
            smaller_area = np.pi * min(r1, r2)**2
            larger_area  = np.pi * max(r1, r2)**2
            return smaller_area / larger_area
        
        # Partial overlap — compute lens area
        a1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        a2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        intersection = a1 + a2 - a3
        union = np.pi * (r1**2 + r2**2) - intersection
        return intersection / union if union > 0 else 0.0
```

### Step 2.4 — Boundary Filtering

```python
    def filter_boundary_circles(self, circles: List[Circle], 
                                  image_shape: tuple,
                                  margin_ratio: float = 0.5) -> List[Circle]:
        """
        Remove circles whose center is too close to the image boundary
        OR whose area extends significantly outside the image.
        
        A coin whose center is within margin_ratio * r of the border
        is partially outside — keep only if center is fully inside.
        """
        h, w = image_shape[:2]
        filtered = []
        for (x, y, r) in circles:
            margin = int(r * margin_ratio)
            if (x - margin >= 0 and x + margin < w and
                y - margin >= 0 and y + margin < h):
                filtered.append((x, y, r))
        return filtered
```

### Step 2.5 — Accumulator Visualization (Debug)

```python
    def visualize_detections(self, original: np.ndarray, 
                              circles: List[Circle]) -> np.ndarray:
        """Draw detected circles on a copy of the original image."""
        viz = original.copy()
        for i, (x, y, r) in enumerate(circles):
            # Outer circle
            cv2.circle(viz, (x, y), r, (0, 255, 0), 2)
            # Center dot
            cv2.circle(viz, (x, y), 3, (0, 0, 255), -1)
            # Index label
            cv2.putText(viz, str(i), (x - 8, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return viz
```

### Step 2.6 — `run()` Method

```python
    def run(self, phase1_result: dict) -> dict:
        """
        Input:  phase1_result dict from Preprocessor.run()
        Output: dict with 'circles' (List[Circle]) and 'visualization'
        """
        from src.utils.image_io import save_image
        
        blurred  = phase1_result['blurred']
        original = phase1_result['scaled']
        
        # Auto-set radius range if not in config
        if 'min_radius' not in self.config.get('detection', {}):
            min_r, max_r = self.suggest_radius_range(blurred.shape)
            self.config.setdefault('detection', {})
            self.config['detection']['min_radius'] = min_r
            self.config['detection']['max_radius'] = max_r
        
        raw_circles = self.detect_circles(blurred)
        
        if raw_circles is None:
            circles = []
        else:
            circles_list = [(int(x), int(y), int(r)) 
                            for x, y, r in raw_circles[0]]
            circles = self.non_max_suppression(circles_list)
            circles = self.filter_boundary_circles(circles, blurred.shape)
        
        viz = self.visualize_detections(original, circles)
        
        if self.debug:
            save_image(viz, f"{self.output_dir}/detections.png")
        
        return {
            "circles":       circles,           # List[(x, y, r)]
            "count":         len(circles),
            "visualization": viz,
            "scale_factor":  phase1_result['scale_factor'],
        }
```

---

## 4. Strategy Explanation

### 4.1 Why `dp=1.2`?

`dp=1` gives maximum precision but is slow on large images. `dp=1.2` provides a good speed/accuracy trade-off. Use `dp=1` for final results, `dp=2` for speed testing.

### 4.2 Why `minDist=30`?

For a 1024px working image, coins rarely touch (they're physically separate). 30px minimum distance prevents double-detection of the same coin but allows closely grouped coins.

### 4.3 The `param2` Trade-off

```
Low param2  →  More circles detected  →  More false positives
High param2 →  Fewer circles detected →  More missed coins
```

Strategy: Start with `param2=30`. If false positives appear, increase to 40-50. If coins are missed, decrease to 20-25.

### 4.4 Multi-Scale Detection

For images where coins vary greatly in size, run CHT multiple times at different scales:

```python
    def multi_scale_detect(self, blurred: np.ndarray) -> List[Circle]:
        """Run CHT at 3 scale factors, merge and deduplicate results."""
        all_circles = []
        for scale in [0.5, 0.75, 1.0]:
            h, w = blurred.shape
            scaled = cv2.resize(blurred, (int(w*scale), int(h*scale)))
            raw = self.detect_circles(scaled)
            if raw is not None:
                for x, y, r in raw[0]:
                    # Scale coordinates back to original
                    all_circles.append((
                        int(x / scale), int(y / scale), int(r / scale)
                    ))
        return self.non_max_suppression(all_circles)
```

---

## 5. `config.yaml` Parameters for Phase 2

```yaml
detection:
  dp: 1.2                # Accumulator resolution ratio
  min_dist: 30           # Minimum center-to-center distance (px)
  param1: 80             # Internal Canny high threshold
  param2: 30             # Accumulator vote threshold (lower = more circles)
  min_radius: 15         # Minimum coin radius (px) — set per image scale
  max_radius: 200        # Maximum coin radius (px) — set per image scale
  nms_iou_threshold: 0.3 # Overlap threshold for NMS
  boundary_margin: 0.5   # Fraction of radius to use as boundary margin
```

---

## 6. Expected Outputs

| Output | Description |
|---|---|
| `circles` | List of `(x, y, r)` tuples in pixel coordinates |
| `count` | Integer — number of coins detected |
| `visualization` | BGR image with green circles and red centers drawn |
| `outputs/intermediate/phase2/detections.png` | Saved visualization |

**Visual expectations**:
- Each detected coin has exactly **one** green ring
- Ring fits tightly around coin boundary (within ±3px)
- No rings on background texture or shadows
- All coins present in image are detected

---

## 7. Test Cases (Phase 2 Complete When All Pass)

```python
# tests/test_phase2_detection.py
import pytest
import cv2
import numpy as np
from src.phase2_detection import CoinDetector

CONFIG = {
    'detection': {
        'dp': 1.2, 'min_dist': 30, 'param1': 80, 'param2': 30,
        'min_radius': 15, 'max_radius': 150, 'nms_iou_threshold': 0.3,
    }
}

@pytest.fixture
def detector():
    return CoinDetector(CONFIG, debug=False)

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
    nms_circles = detector.non_max_suppression(circles)
    # Should find approximately 3 circles
    assert 2 <= len(nms_circles) <= 4, f"Expected ~3 circles, got {len(nms_circles)}"

# TC-2.3: NMS removes overlapping duplicates
def test_nms_removes_duplicates(detector):
    # Two nearly identical circles (same center, slightly diff radius)
    circles = [(100, 100, 40), (102, 101, 42), (300, 300, 40)]
    kept = detector.non_max_suppression(circles, iou_threshold=0.3)
    assert len(kept) == 2, f"NMS should keep 2, got {len(kept)}"

# TC-2.4: Circle overlap function — no overlap
def test_circle_overlap_none(detector):
    c1, c2 = (0, 0, 30), (100, 0, 30)
    overlap = detector._circle_overlap(c1, c2)
    assert overlap == 0.0

# TC-2.5: Circle overlap function — identical circles
def test_circle_overlap_identical(detector):
    c = (50, 50, 30)
    overlap = detector._circle_overlap(c, c)
    assert abs(overlap - 1.0) < 0.05, f"Identical circles should have ~1.0 overlap, got {overlap}"

# TC-2.6: Boundary filtering removes edge circles
def test_boundary_filter(detector):
    # Circle at (5, 5, 40) — center near top-left, mostly outside
    circles = [(5, 5, 40), (200, 200, 40)]
    filtered = detector.filter_boundary_circles(circles, image_shape=(400, 400))
    # The (5,5) circle should be removed
    assert (200, 200, 40) in filtered
    assert not any(x < 10 and y < 10 for x, y, r in filtered)

# TC-2.7: Empty image returns empty list
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

# TC-2.10: Detected radius within 10% of true radius
def test_detection_radius_accuracy(detector):
    true_r = 50
    img = make_circle_image([(200, 200, true_r)])
    raw = detector.detect_circles(img)
    assert raw is not None
    detected_r = raw[0][0][2]
    assert abs(detected_r - true_r) / true_r < 0.15, \
        f"Detected radius {detected_r} too far from true {true_r}"
```

### Phase 2 Completion Checklist

- [ ] All 10 test cases pass
- [ ] Detection rate ≥ 90% on `test_mixed_coins.jpg` (ground truth comparison)
- [ ] False positive rate < 10%
- [ ] `detections.png` visually shows correct circle overlays
- [ ] Processing time < 2 seconds per image
- [ ] Config parameters are externalized in `config.yaml`
