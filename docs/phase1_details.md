# Phase 1: Setup & Image Preprocessing
## Coin Counting System — Classical CV Pipeline

---

## 1. Overview

Preprocessing is the **foundation** of the entire pipeline. Poorly preprocessed images cause false detections and missed coins downstream. This phase transforms a raw photograph into a clean, noise-reduced representation optimized for the Circular Hough Transform in Phase 2.

**Goal**: Convert raw input → normalized grayscale + edge map ready for CHT.

---

## 2. Implementation Steps

### Step 1.1 — Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify OpenCV
python -c "import cv2; print(cv2.__version__)"  # Expected: 4.9.x
```

Create the project scaffold:
```bash
mkdir -p src/utils outputs/intermediate/phase1 data/sample_images data/ground_truth tests docs
touch src/__init__.py src/utils/__init__.py
```

### Step 1.2 — Image Loading & Validation

```python
# src/utils/image_io.py
import cv2
import numpy as np
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def load_image(path: str) -> np.ndarray:
    """Load image from disk, validate it is non-empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported format: {p.suffix}")
    
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    if img.size == 0:
        raise ValueError("Image is empty (0 bytes of pixel data)")
    return img

def save_image(img: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)
```

### Step 1.3 — Grayscale Conversion

```python
# src/phase1_preprocessing.py
import cv2
import numpy as np

class Preprocessor:
    def __init__(self, config: dict, debug: bool = False, output_dir: str = "outputs/intermediate/phase1"):
        self.config = config
        self.debug = debug
        self.output_dir = output_dir

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale. Coins have circular shape best captured in intensity."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

**Why grayscale?** CHT operates on intensity gradients. Color information is preserved in the original image for Phase 4 feature extraction.

### Step 1.4 — CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhance local contrast without blowing out highlights.
        CLAHE divides the image into tiles and equalizes each tile's
        histogram independently, preventing over-amplification of noise.
        """
        clip_limit    = self.config.get('clahe_clip_limit', 2.0)
        tile_grid_size = self.config.get('clahe_tile_size', (8, 8))
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)
```

**Why CLAHE?** Coins photographed under uneven lighting will have washed-out or dark regions. CLAHE ensures edges remain sharp even in bright/dark zones, improving CHT accuracy significantly.

### Step 1.5 — Gaussian Blur (Noise Reduction)

```python
    def gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Smooth high-frequency noise before edge detection.
        Kernel size must be odd. Larger kernels = more smoothing.
        Typical: 5×5 for clean images, 9×9 for noisy ones.
        """
        kernel_size = self.config.get('blur_kernel', (5, 5))
        sigma       = self.config.get('blur_sigma', 1.5)
        return cv2.GaussianBlur(gray, kernel_size, sigma)
```

**Trade-off**: Too little blur → CHT detects noise as circles. Too much blur → thin coin edges disappear. Kernel of (5,5) σ=1.5 is a robust starting point.

### Step 1.6 — Canny Edge Detection

```python
    def canny_edges(self, blurred: np.ndarray) -> np.ndarray:
        """
        Detect edges using the Canny algorithm:
        1. Compute gradient magnitude/direction (Sobel)
        2. Non-maximum suppression (thin edges to 1px)
        3. Double threshold + hysteresis (remove weak edges)
        
        Auto-thresholding using Otsu's method for adaptability.
        """
        # Auto-compute thresholds from image statistics
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low  = self.config.get('canny_low',  otsu_thresh * 0.5)
        high = self.config.get('canny_high', otsu_thresh)
        
        return cv2.Canny(blurred, low, high)
```

**Why auto-thresholds?** Manual thresholds fail under different lighting conditions. Otsu-derived thresholds adapt to each image's histogram.

### Step 1.7 — Scale Normalization (Optional)

```python
    def normalize_scale(self, image: np.ndarray) -> np.ndarray:
        """
        Resize very large images to a standard working resolution.
        This keeps CHT radius ranges consistent across cameras.
        Target: longest side = 1024px (preserves aspect ratio).
        """
        max_dim = self.config.get('max_working_dim', 1024)
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
```

### Step 1.8 — `run()` Method (Pipeline Glue)

```python
    def run(self, image_path: str) -> dict:
        """Execute full preprocessing pipeline, return all intermediate results."""
        from src.utils.image_io import load_image, save_image
        
        original = load_image(image_path)
        scaled   = self.normalize_scale(original)
        gray     = self.to_grayscale(scaled)
        clahe    = self.apply_clahe(gray)
        blurred  = self.gaussian_blur(clahe)
        edges    = self.canny_edges(blurred)
        
        if self.debug:
            save_image(gray,    f"{self.output_dir}/gray.png")
            save_image(clahe,   f"{self.output_dir}/clahe.png")
            save_image(blurred, f"{self.output_dir}/blurred.png")
            save_image(edges,   f"{self.output_dir}/edges.png")
        
        return {
            "original":  original,
            "scaled":    scaled,
            "gray":      gray,
            "clahe":     clahe,
            "blurred":   blurred,
            "edges":     edges,
            "scale_factor": scaled.shape[1] / original.shape[1]
        }
```

---

## 3. Strategy Explanation

### Why This Order Matters

```
Raw image → Grayscale → CLAHE → Gaussian Blur → Canny
```

1. **Grayscale first**: Eliminates color channel noise; CHT needs single-channel input.
2. **CLAHE before blur**: Enhance contrast on the original detail; blurring afterwards won't destroy the enhanced edges.
3. **Blur before Canny**: Canny's Sobel gradient is noise-sensitive — blurring prevents salt-and-pepper noise from being detected as edges.
4. **Adaptive Canny**: Robustness across varied lighting conditions.

### Failure Modes & Mitigations

| Failure | Symptom | Mitigation |
|---|---|---|
| Over-exposed image | Coins merge with background | CLAHE recovers local contrast |
| Very noisy image | Canny finds thousands of edges | Increase blur kernel to (9,9) |
| Image too large | Slow processing | Scale to max 1024px |
| Dark background | Low contrast edges | CLAHE + lower Canny thresholds |

---

## 4. `config.yaml` Parameters for Phase 1

```yaml
preprocessing:
  max_working_dim: 1024       # Max pixels on longest side
  clahe_clip_limit: 2.0       # Higher = more contrast enhancement
  clahe_tile_size: [8, 8]     # Grid size for CLAHE tiles
  blur_kernel: [5, 5]         # Gaussian blur kernel (must be odd)
  blur_sigma: 1.5             # Gaussian sigma
  canny_low_ratio: 0.5        # Canny low = Otsu * this ratio
  canny_high_ratio: 1.0       # Canny high = Otsu * this ratio
```

---

## 5. Expected Outputs

| Output | Description |
|---|---|
| `outputs/intermediate/phase1/gray.png` | Grayscale version of scaled image |
| `outputs/intermediate/phase1/clahe.png` | Contrast-enhanced grayscale |
| `outputs/intermediate/phase1/blurred.png` | Smoothed image |
| `outputs/intermediate/phase1/edges.png` | Binary edge map (white edges on black) |
| Return dict | `{original, scaled, gray, clahe, blurred, edges, scale_factor}` |

**Visual expectations**:
- `edges.png` should show **clean, continuous rings** for each coin
- No fragmented or broken circles
- Background noise should be minimal (sparse white dots at most)

---

## 6. Test Cases (Phase 1 Complete When All Pass)

```python
# tests/test_phase1_preprocessing.py
import pytest
import cv2
import numpy as np
from src.phase1_preprocessing import Preprocessor

CONFIG = {
    'blur_kernel': (5, 5),
    'blur_sigma': 1.5,
    'clahe_clip_limit': 2.0,
    'clahe_tile_size': (8, 8),
    'max_working_dim': 1024,
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

# TC-1.3: Gaussian blur produces non-identical but same-shape output
def test_blur_smooths_image(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    blurred = preprocessor.gaussian_blur(gray)
    assert blurred.shape == gray.shape
    assert not np.array_equal(blurred, gray), "Blur should change pixel values"

# TC-1.4: Canny output is binary (0 or 255 only)
def test_canny_is_binary(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    blurred = preprocessor.gaussian_blur(gray)
    edges = preprocessor.canny_edges(blurred)
    unique_vals = np.unique(edges)
    assert set(unique_vals).issubset({0, 255}), f"Canny has non-binary values: {unique_vals}"

# TC-1.5: Canny detects edges on synthetic circle image
def test_canny_detects_circle_edges(preprocessor, sample_bgr):
    gray = preprocessor.to_grayscale(sample_bgr)
    blurred = preprocessor.gaussian_blur(gray)
    edges = preprocessor.canny_edges(blurred)
    # Should have significant white pixels (circle boundary detected)
    white_pixel_ratio = np.sum(edges > 0) / edges.size
    assert white_pixel_ratio > 0.01, "Canny found no edges on synthetic circle"
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
        from src.utils.image_io import load_image
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
```

### Phase 1 Completion Checklist

- [ ] All 10 test cases pass (`pytest tests/test_phase1_preprocessing.py -v`)
- [ ] `outputs/intermediate/phase1/edges.png` shows clean coin rings on test image
- [ ] Processing time for 1080p image < 1 second
- [ ] No errors on all 5 sample images in `data/sample_images/`
- [ ] Config parameters documented in `config.yaml`
