# Phase 3: Coin Segmentation
## Coin Counting System — Classical CV Pipeline

---

## 1. Overview

Once circles are detected, each coin must be **isolated** from the image for individual analysis. Segmentation extracts each coin as a separate image region (ROI), applies a circular mask to exclude background pixels, and normalizes the result to a standard size for consistent feature extraction in Phase 4.

**Goal**: For each `(x, y, r)` from Phase 2 → produce a list of normalized 128×128 BGR coin images with circular masks.

---

## 2. Implementation Steps

### Step 3.1 — ROI Extraction

```python
# src/phase3_segmentation.py
import cv2
import numpy as np
from typing import List, Tuple, Dict

Circle = Tuple[int, int, int]

class CoinSegmenter:
    def __init__(self, config: dict, debug: bool = False,
                 output_dir: str = "outputs/intermediate/phase3"):
        self.config     = config
        self.debug      = debug
        self.output_dir = output_dir
        self.target_size = config.get('target_size', 128)
        self.padding_ratio = config.get('padding_ratio', 0.05)

    def crop_roi(self, image: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
        """
        Extract a square bounding box around the coin with optional padding.
        
        Padding prevents masking artifacts at exact circle boundary by
        including a small margin (5% of radius by default).
        
        Clamps to image boundaries to handle partially out-of-frame coins.
        """
        pad = int(r * self.padding_ratio)
        r_padded = r + pad
        
        # Compute bounding box with clamping
        x1 = max(0, x - r_padded)
        y1 = max(0, y - r_padded)
        x2 = min(image.shape[1], x + r_padded)
        y2 = min(image.shape[0], y + r_padded)
        
        return image[y1:y2, x1:x2].copy(), (x1, y1)  # ROI + top-left offset
```

### Step 3.2 — Circular Mask Application

```python
    def apply_circular_mask(self, roi: np.ndarray, 
                             cx_local: int, cy_local: int, 
                             r: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a circular binary mask to isolate the coin from background.
        
        Returns:
            masked_roi: BGR image with background set to black (0,0,0)
            mask:       Binary mask (255=coin, 0=background)
        
        Why not use the raw rectangle?
        - Background pixels contaminate color statistics
        - Sharp corners introduce non-coin hues in feature histograms
        - Circular mask mirrors the actual coin shape exactly
        """
        h, w = roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw filled white circle at local coordinates
        cv2.circle(mask, (cx_local, cy_local), r, 255, -1)
        
        # Optional: apply slight erosion to avoid boundary pixels
        erode_px = self.config.get('mask_erosion_px', 2)
        if erode_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (erode_px*2+1, erode_px*2+1))
            mask = cv2.erode(mask, kernel)
        
        # Apply mask: background becomes pure black
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        return masked_roi, mask
```

### Step 3.3 — Normalization to Standard Canvas

```python
    def normalize_roi(self, masked_roi: np.ndarray, 
                       mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize coin ROI to target_size × target_size.
        
        Using INTER_AREA for downscaling (antialiasing), INTER_CUBIC for 
        upscaling (smooth interpolation).
        
        The mask is also resized to match, ensuring it can be reused in
        Phase 4 for masked color statistics.
        """
        h, w = masked_roi.shape[:2]
        t = self.target_size
        
        interp = cv2.INTER_AREA if max(h, w) > t else cv2.INTER_CUBIC
        
        norm_roi  = cv2.resize(masked_roi, (t, t), interpolation=interp)
        norm_mask = cv2.resize(mask,       (t, t), interpolation=cv2.INTER_NEAREST)
        
        return norm_roi, norm_mask
```

### Step 3.4 — Per-Coin Metadata

```python
    def segment_coin(self, image: np.ndarray, 
                     circle: Circle, 
                     coin_index: int) -> Dict:
        """
        Full segmentation pipeline for a single coin.
        Returns a metadata dict containing all representations.
        """
        x, y, r = circle
        
        # Step 1: Extract ROI
        roi, (x1, y1) = self.crop_roi(image, x, y, r)
        
        # Step 2: Compute local center in cropped ROI coordinate system
        cx_local = x - x1
        cy_local = y - y1
        
        # Step 3: Apply circular mask
        masked_roi, mask = self.apply_circular_mask(roi, cx_local, cy_local, r)
        
        # Step 4: Normalize
        norm_roi, norm_mask = self.normalize_roi(masked_roi, mask)
        
        return {
            "index":       coin_index,
            "circle":      circle,           # (x, y, r) in full image coords
            "roi":         roi,              # Raw cropped square
            "masked_roi":  masked_roi,       # Circle-masked coin
            "mask":        mask,             # Binary mask
            "norm_roi":    norm_roi,         # 128×128 normalized coin
            "norm_mask":   norm_mask,        # 128×128 normalized mask
            "local_center": (cx_local, cy_local),
        }
```

### Step 3.5 — Batch Segmentation + Debug Output

```python
    def run(self, phase1_result: dict, phase2_result: dict) -> dict:
        """
        Segment all detected coins.
        Input:  Phase 1 result (for original color image) + Phase 2 result (circles)
        Output: List of coin segment dicts
        """
        from src.utils.image_io import save_image
        import os
        
        image   = phase1_result['scaled']   # Full color image
        circles = phase2_result['circles']  # List[(x, y, r)]
        
        segments = []
        for i, circle in enumerate(circles):
            seg = self.segment_coin(image, circle, i)
            segments.append(seg)
            
            if self.debug:
                os.makedirs(f"{self.output_dir}/coins", exist_ok=True)
                save_image(seg['norm_roi'],  
                           f"{self.output_dir}/coins/coin_{i:03d}_norm.png")
                save_image(seg['masked_roi'], 
                           f"{self.output_dir}/coins/coin_{i:03d}_masked.png")
        
        if self.debug:
            grid = self._make_coin_grid(segments)
            save_image(grid, f"{self.output_dir}/all_coins_grid.png")
        
        return {
            "segments":   segments,
            "count":      len(segments),
        }

    def _make_coin_grid(self, segments: list, cols: int = 8) -> np.ndarray:
        """Arrange all normalized coins into a grid image for overview."""
        t = self.target_size
        n = len(segments)
        rows = (n + cols - 1) // cols
        
        grid = np.zeros((rows * t, cols * t, 3), dtype=np.uint8)
        for i, seg in enumerate(segments):
            r, c = divmod(i, cols)
            grid[r*t:(r+1)*t, c*t:(c+1)*t] = seg['norm_roi']
        
        return grid
```

---

## 3. Strategy Explanation

### 3.1 Why a Circular Mask?

Without masking, the square ROI includes rectangular corners that contain background. If the coin is on a dark surface, those corners are harmless. But on colored backgrounds, corner pixels pollute color histograms leading to misclassification.

```
WITHOUT mask:        WITH mask:
┌──────────┐         ┌──────────┐
│ bg  /    │         │    /     │
│    /coin │   →     │   coin   │
│   coin \ │         │        \ │
│ bg       │         │          │
└──────────┘         └──────────┘
  Corners pollute       Pure coin pixels
  color stats           only
```

### 3.2 Mask Erosion

Eroding the circular mask by 2px removes the outer ring of boundary pixels, which may contain blended background colors (especially on natural backgrounds). This produces cleaner color statistics.

### 3.3 Why 128×128 Normalization?

- **Consistent feature space**: Feature vectors computed in Phase 4 assume fixed-size input
- **Computational efficiency**: Small enough for fast histogram computation
- **Sufficient detail**: 128px captures coin color patterns clearly
- Configurable via `target_size` parameter

### 3.4 Handling Partially Visible Coins

For coins near the image border, the ROI is clamped and the mask is recomputed. The local center coordinates are adjusted accordingly. These partial coins are flagged for low-confidence classification in Phase 5.

---

## 4. `config.yaml` Parameters for Phase 3

```yaml
segmentation:
  target_size: 128      # Normalized coin image size (px)
  padding_ratio: 0.05   # ROI padding fraction of radius
  mask_erosion_px: 2    # Erode mask by this many pixels (0 to disable)
```

---

## 5. Expected Outputs

| Output | Description |
|---|---|
| `segments` | List of dicts, one per coin, with all image representations |
| `outputs/intermediate/phase3/coins/coin_000_norm.png` | Normalized 128×128 coin image |
| `outputs/intermediate/phase3/coins/coin_000_masked.png` | Masked raw coin crop |
| `outputs/intermediate/phase3/all_coins_grid.png` | Overview grid of all coins |

**Visual expectations**:
- Each `coin_XXX_norm.png` shows a single coin centered on black background
- Coin fills most of the 128×128 canvas (radius ≈ 60px)
- No visible background colors from the table/surface
- Grid image provides quick QA overview of all detected coins

---

## 6. Test Cases (Phase 3 Complete When All Pass)

```python
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
```

### Phase 3 Completion Checklist

- [ ] All 10 test cases pass
- [ ] `all_coins_grid.png` shows all coins cleanly isolated on black background
- [ ] No visible table/background color contamination in any coin ROI
- [ ] Normalized coins are centered (coin fills ~90% of canvas)
- [ ] Partially visible coins are handled without crash
- [ ] Processing time < 0.5 seconds for 20 coins
