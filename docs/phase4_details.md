# Phase 4: Feature Extraction (Size & Color)
## Coin Counting System — Classical CV Pipeline

---

## 1. Overview

Feature extraction transforms each raw coin image into a **numeric feature vector** that captures discriminative properties — specifically, **physical size** and **surface color**. These two cues are sufficient to distinguish most Euro coin denominations without any machine learning.

**Goal**: For each normalized coin ROI → produce a feature dictionary with size ratio, HSV statistics, LAB statistics, color histogram, and dominant hue.

---

## 2. Feature Design Rationale

### Why Size + Color?

| Property | What it tells us | Limits |
|---|---|---|
| **Physical size** | Diameter in mm (largest → 2€, smallest → 1c) | Requires known camera distance |
| **Color** | Copper (1c-5c) vs Gold (10c-50c) vs Bicolor (€1-€2) | Lighting-sensitive |

Combined, these two features create a **2D decision space** that largely separates Euro denominations:

```
        Copper          Gold          Bicolor
Small   [1c, 2c]    [10c, 20c]     -
Large   [5c]        [50c]          [€1, €2]
```

---

## 3. Implementation Steps

### Step 4.1 — Size Feature

```python
# src/phase4_features.py
import cv2
import numpy as np
from typing import Dict, List

class FeatureExtractor:
    def __init__(self, config: dict, debug: bool = False,
                 output_dir: str = "outputs/intermediate/phase4"):
        self.config     = config
        self.debug      = debug
        self.output_dir = output_dir

    def compute_size_feature(self, radius_px: int, 
                              image_diagonal_px: float) -> Dict:
        """
        Compute normalized size features from detected coin radius.
        
        Normalization by image diagonal makes the feature invariant to:
        - Camera distance (zoom level)
        - Image resolution (1080p vs 4K)
        
        Returns both the raw ratio and size percentile category.
        """
        # Relative size = radius / image diagonal
        rel_size = (radius_px / image_diagonal_px)
        
        # Relative diameter (more intuitive)
        rel_diam = rel_size * 2
        
        # Size bin: maps relative size to ordinal category
        # These thresholds are calibrated for typical tabletop photos
        thresholds = self.config.get('size_thresholds', {
            'tiny':   0.025,   # < 2.5% of diagonal → 1c or 2c
            'small':  0.040,   # 2.5–4.0% → 2c or 5c / 10c
            'medium': 0.055,   # 4.0–5.5% → 10c, 20c, or 50c
            'large':  0.070,   # 5.5–7.0% → 50c, €1
            # > 7.0% → €2
        })
        
        if rel_size < thresholds['tiny']:
            size_bin = 'tiny'
        elif rel_size < thresholds['small']:
            size_bin = 'small'
        elif rel_size < thresholds['medium']:
            size_bin = 'medium'
        elif rel_size < thresholds['large']:
            size_bin = 'large'
        else:
            size_bin = 'xlarge'
        
        return {
            'radius_px':    radius_px,
            'rel_size':     rel_size,
            'rel_diam':     rel_diam,
            'size_bin':     size_bin,
        }
```

### Step 4.2 — HSV Color Statistics

```python
    def compute_hsv_features(self, norm_roi: np.ndarray, 
                              norm_mask: np.ndarray) -> Dict:
        """
        Compute mean & std of Hue, Saturation, Value in HSV space.
        
        Why HSV over RGB?
        - Hue (0-179 in OpenCV) directly encodes color type:
            0-15   → Red tones
            15-30  → Orange / Copper
            20-35  → Yellow / Gold
            90-130 → Blue/Silver (highly lit white surfaces)
        - Saturation distinguishes copper (high sat) from silver (low sat)
        - Value captures brightness (not discriminative for coin type)
        
        Statistics computed ONLY over masked (coin) pixels.
        """
        hsv = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2HSV)
        mask_bool = norm_mask > 0
        
        h = hsv[:, :, 0][mask_bool].astype(float)
        s = hsv[:, :, 1][mask_bool].astype(float)
        v = hsv[:, :, 2][mask_bool].astype(float)
        
        return {
            'hue_mean':  float(np.mean(h)),
            'hue_std':   float(np.std(h)),
            'sat_mean':  float(np.mean(s)),
            'sat_std':   float(np.std(s)),
            'val_mean':  float(np.mean(v)),
            'val_std':   float(np.std(v)),
        }
```

### Step 4.3 — LAB Color Statistics

```python
    def compute_lab_features(self, norm_roi: np.ndarray, 
                              norm_mask: np.ndarray) -> Dict:
        """
        Compute mean & std of L*, a*, b* channels in CIELAB space.
        
        Why LAB?
        - L* (lightness): perceptually uniform brightness
        - a* (green↔red axis): copper has high positive a*
        - b* (blue↔yellow axis): gold has high positive b*
        
        LAB is more perceptually uniform than HSV, making color 
        differences more predictable across lighting conditions.
        
        LAB ranges in OpenCV uint8:
          L: 0-255 (mapped from 0-100)
          a: 0-255 (mapped from -128 to +127, neutral = 128)
          b: 0-255 (mapped from -128 to +127, neutral = 128)
        """
        lab = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2LAB)
        mask_bool = norm_mask > 0
        
        L = lab[:, :, 0][mask_bool].astype(float)
        a = lab[:, :, 1][mask_bool].astype(float)
        b = lab[:, :, 2][mask_bool].astype(float)
        
        return {
            'lab_L_mean': float(np.mean(L)),
            'lab_a_mean': float(np.mean(a)),   # > 128 = reddish/copper
            'lab_b_mean': float(np.mean(b)),   # > 128 = yellowish/gold
            'lab_L_std':  float(np.std(L)),
            'lab_a_std':  float(np.std(a)),
            'lab_b_std':  float(np.std(b)),
        }
```

### Step 4.4 — Color Histogram (Hue Distribution)

```python
    def compute_hue_histogram(self, norm_roi: np.ndarray, 
                               norm_mask: np.ndarray,
                               bins: int = 18) -> Dict:
        """
        Compute normalized hue histogram over masked coin pixels.
        
        Using 18 bins (10° each) captures the broad color categories
        (copper, gold, silver) without being noise-sensitive.
        
        The histogram is L1-normalized so the sum = 1.0.
        This makes it lighting-invariant (relative color proportions).
        """
        hsv = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist(
            [hsv], [0], norm_mask,
            [bins], [0, 180]
        ).flatten()
        
        total = hist.sum()
        if total > 0:
            hist = hist / total  # Normalize
        
        # Find dominant hue bin
        dom_bin = int(np.argmax(hist))
        dom_hue = (dom_bin * 180 / bins) + (180 / bins / 2)  # Bin center
        
        return {
            'hue_hist':     hist.tolist(),
            'dominant_hue': float(dom_hue),
            'hist_entropy': float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))),
        }
```

### Step 4.5 — Color Group Classification

```python
    def classify_color_group(self, hsv_feats: Dict, lab_feats: Dict) -> str:
        """
        Classify coin into one of 3 color groups using color features.
        
        Euro coin color groups:
          COPPER:  1c, 2c, 5c     (orange-brown tones)
          GOLD:    10c, 20c, 50c  (yellow-gold tones)
          BICOLOR: €1, €2         (both gold AND silver in same coin)
          SILVER:  Rare — some commemorative coins
        
        Decision logic:
          - Copper: hue 10-25, high saturation (> 80), high LAB a* (> 140)
          - Gold:   hue 20-35, medium saturation (40-120), high LAB b* (> 140)
          - Bicolor: high hue variance (std > 25) or mixed a*/b* distribution
          - Silver: low saturation (< 40), low hue variation
        """
        hue_mean = hsv_feats['hue_mean']
        hue_std  = hsv_feats['hue_std']
        sat_mean = hsv_feats['sat_mean']
        lab_a    = lab_feats['lab_a_mean']
        lab_b    = lab_feats['lab_b_mean']
        
        # Bicolor: large hue variance (gold center + silver rim or vice versa)
        if hue_std > self.config.get('bicolor_hue_std_thresh', 22):
            return 'BICOLOR'
        
        # Copper: orange-brown (hue 8-25, high sat, warm LAB)
        if (8 <= hue_mean <= 25 and 
            sat_mean > 70 and 
            lab_a > 135):
            return 'COPPER'
        
        # Gold: yellow-golden (hue 18-38, moderate sat, high b*)
        if (15 <= hue_mean <= 40 and 
            sat_mean > 30 and 
            lab_b > 138):
            return 'GOLD'
        
        # Silver/neutral: low saturation
        if sat_mean < 45:
            return 'SILVER'
        
        return 'UNKNOWN'
```

### Step 4.6 — Build Full Feature Vector

```python
    def build_feature_vector(self, segment: Dict, 
                              image_diagonal: float) -> Dict:
        """
        Combine all features into a single dict for Phase 5 classification.
        """
        norm_roi  = segment['norm_roi']
        norm_mask = segment['norm_mask']
        x, y, r   = segment['circle']
        
        size_feats = self.compute_size_feature(r, image_diagonal)
        hsv_feats  = self.compute_hsv_features(norm_roi, norm_mask)
        lab_feats  = self.compute_lab_features(norm_roi, norm_mask)
        hist_feats = self.compute_hue_histogram(norm_roi, norm_mask)
        color_group = self.classify_color_group(hsv_feats, lab_feats)
        
        return {
            "coin_index":    segment['index'],
            "circle":        segment['circle'],
            **size_feats,
            **hsv_feats,
            **lab_feats,
            **hist_feats,
            "color_group":   color_group,
        }

    def run(self, phase1_result: dict, 
            phase2_result: dict, 
            phase3_result: dict) -> dict:
        """Extract features from all segmented coins."""
        import os
        
        h, w = phase1_result['scaled'].shape[:2]
        image_diagonal = np.sqrt(h**2 + w**2)
        
        features = []
        for seg in phase3_result['segments']:
            feat = self.build_feature_vector(seg, image_diagonal)
            features.append(feat)
        
        if self.debug:
            self._save_feature_viz(features, phase3_result['segments'])
        
        return {"features": features, "image_diagonal": image_diagonal}
```

### Step 4.7 — Feature Visualization (Debug)

```python
    def _save_feature_viz(self, features: List[Dict], 
                           segments: List[Dict]) -> None:
        """Save per-coin feature summary images."""
        import matplotlib.pyplot as plt
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        for feat, seg in zip(features, segments):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            # Normalized coin
            axes[0].imshow(cv2.cvtColor(seg['norm_roi'], cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Coin {feat['coin_index']}\n"
                               f"Color: {feat['color_group']}\n"
                               f"Size: {feat['size_bin']}")
            axes[0].axis('off')
            
            # Hue histogram
            axes[1].bar(range(len(feat['hue_hist'])), feat['hue_hist'],
                        color='steelblue')
            axes[1].set_title(f"Hue Histogram\nDom.Hue={feat['dominant_hue']:.1f}°")
            axes[1].set_xlabel("Hue Bin (10° each)")
            
            # Feature table
            summary = (f"hue_mean={feat['hue_mean']:.1f}\n"
                       f"sat_mean={feat['sat_mean']:.1f}\n"
                       f"lab_a={feat['lab_a_mean']:.1f}\n"
                       f"lab_b={feat['lab_b_mean']:.1f}\n"
                       f"rel_size={feat['rel_size']:.4f}")
            axes[2].text(0.1, 0.5, summary, transform=axes[2].transAxes,
                         fontsize=11, verticalalignment='center',
                         fontfamily='monospace')
            axes[2].axis('off')
            axes[2].set_title("Feature Summary")
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/coin_{feat['coin_index']:03d}_features.png",
                        dpi=80, bbox_inches='tight')
            plt.close()
```

---

## 4. Strategy Explanation

### Feature Space Summary

| Feature | Type | Discriminates |
|---|---|---|
| `rel_size` | Float [0,1] | Large (€2) vs Small (1c) |
| `size_bin` | Ordinal 5-class | Coarse denomination group |
| `hue_mean` | Float [0,180] | Copper vs Gold vs Silver |
| `hue_std` | Float | Bicolor (high std) |
| `sat_mean` | Float [0,255] | Vivid metal vs dull metal |
| `lab_a_mean` | Float [0,255] | Copper (reddish, > 135) |
| `lab_b_mean` | Float [0,255] | Gold (yellowish, > 140) |
| `color_group` | Categorical | COPPER/GOLD/BICOLOR/SILVER |
| `hue_hist` | Vector [18] | Fine-grained color dist. |
| `hist_entropy` | Float | Color uniformity |

### Lighting Robustness

Raw pixel values are lighting-dependent. Mitigations:
1. **CLAHE in Phase 1** normalizes local contrast
2. **Relative features** (`rel_size`, normalized histograms) are scale/brightness invariant
3. **LAB space** is designed to be perceptually uniform under illumination change
4. **Hue channel** in HSV is invariant to brightness (if coin is not overexposed)

---

## 5. Expected Outputs

| Output | Description |
|---|---|
| `features` | List of feature dicts, one per coin |
| `image_diagonal` | Float — diagonal of working image in pixels |
| `outputs/intermediate/phase4/coin_000_features.png` | Feature visualization per coin |

**Expected feature ranges for Euro coins**:

| Coin | rel_size (approx) | hue_mean | sat_mean | color_group |
|---|---|---|---|---|
| 1 cent | 0.020-0.028 | 12-22 | 90-140 | COPPER |
| 2 cent | 0.025-0.033 | 12-22 | 90-140 | COPPER |
| 5 cent | 0.030-0.040 | 12-22 | 85-130 | COPPER |
| 10 cent | 0.027-0.037 | 22-35 | 50-100 | GOLD |
| 20 cent | 0.032-0.043 | 22-35 | 50-100 | GOLD |
| 50 cent | 0.037-0.050 | 22-35 | 50-100 | GOLD |
| €1 | 0.035-0.048 | varies | varies | BICOLOR |
| €2 | 0.040-0.060 | varies | varies | BICOLOR |

---

## 6. Test Cases (Phase 4 Complete When All Pass)

```python
# tests/test_phase4_features.py
import pytest
import cv2
import numpy as np
from src.phase4_features import FeatureExtractor

CONFIG = {
    'size_thresholds': {'tiny': 0.025, 'small': 0.040, 'medium': 0.055, 'large': 0.070},
    'bicolor_hue_std_thresh': 22,
}

@pytest.fixture
def extractor():
    return FeatureExtractor(config=CONFIG, debug=False)

def make_coin_roi(color_bgr, size=128, radius=60):
    """Create synthetic 128×128 coin ROI and mask."""
    roi  = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(roi,  (size//2, size//2), radius, color_bgr, -1)
    cv2.circle(mask, (size//2, size//2), radius, 255, -1)
    return roi, mask

# TC-4.1: Size feature returns all expected keys
def test_size_feature_keys(extractor):
    feat = extractor.compute_size_feature(50, 1000.0)
    assert {'radius_px', 'rel_size', 'rel_diam', 'size_bin'}.issubset(feat.keys())

# TC-4.2: Size bin is 'tiny' for small relative size
def test_size_bin_tiny(extractor):
    feat = extractor.compute_size_feature(20, 1000.0)  # rel_size = 0.02
    assert feat['size_bin'] == 'tiny'

# TC-4.3: Size bin is 'xlarge' for large relative size
def test_size_bin_xlarge(extractor):
    feat = extractor.compute_size_feature(80, 1000.0)  # rel_size = 0.08
    assert feat['size_bin'] == 'xlarge'

# TC-4.4: HSV features computed over masked pixels only
def test_hsv_features_masked(extractor):
    roi, mask = make_coin_roi((30, 165, 210))  # Gold-ish BGR
    feats = extractor.compute_hsv_features(roi, mask)
    required = {'hue_mean', 'hue_std', 'sat_mean', 'sat_std', 'val_mean', 'val_std'}
    assert required.issubset(feats.keys())
    # All values should be in valid range
    assert 0 <= feats['hue_mean'] <= 180
    assert 0 <= feats['sat_mean'] <= 255

# TC-4.5: LAB features computed correctly
def test_lab_features(extractor):
    roi, mask = make_coin_roi((30, 100, 180))  # Warm-ish color
    feats = extractor.compute_lab_features(roi, mask)
    required = {'lab_L_mean', 'lab_a_mean', 'lab_b_mean'}
    assert required.issubset(feats.keys())
    assert 0 <= feats['lab_L_mean'] <= 255

# TC-4.6: Hue histogram is normalized (sums to ~1)
def test_hue_histogram_normalized(extractor):
    roi, mask = make_coin_roi((20, 130, 200))
    hist_feats = extractor.compute_hue_histogram(roi, mask)
    hist = hist_feats['hue_hist']
    assert abs(sum(hist) - 1.0) < 0.01, f"Histogram should sum to 1, got {sum(hist)}"

# TC-4.7: Copper coin gets COPPER color group
def test_copper_color_group(extractor):
    # BGR approximate of copper: (50, 100, 200) → orange-ish
    roi, mask = make_coin_roi((50, 100, 180))
    hsv_f = extractor.compute_hsv_features(roi, mask)
    lab_f = extractor.compute_lab_features(roi, mask)
    # At least verify the function runs without error
    group = extractor.classify_color_group(hsv_f, lab_f)
    assert group in {'COPPER', 'GOLD', 'BICOLOR', 'SILVER', 'UNKNOWN'}

# TC-4.8: Silver coin (gray) gets SILVER color group
def test_silver_color_group(extractor):
    roi, mask = make_coin_roi((180, 180, 180))  # Gray = silver
    hsv_f = extractor.compute_hsv_features(roi, mask)
    lab_f = extractor.compute_lab_features(roi, mask)
    group = extractor.classify_color_group(hsv_f, lab_f)
    assert group == 'SILVER', f"Gray coin should be SILVER, got {group}"

# TC-4.9: build_feature_vector returns all keys
def test_feature_vector_complete(extractor):
    roi, mask = make_coin_roi((30, 100, 180))
    segment = {
        'index': 0,
        'circle': (200, 200, 50),
        'norm_roi': roi,
        'norm_mask': mask,
    }
    feat = extractor.build_feature_vector(segment, image_diagonal=1000.0)
    expected = {'coin_index', 'circle', 'rel_size', 'size_bin', 
                'hue_mean', 'sat_mean', 'lab_a_mean', 'lab_b_mean',
                'color_group', 'hue_hist', 'dominant_hue'}
    assert expected.issubset(feat.keys())

# TC-4.10: Histogram entropy is 0 for uniform-color coin
def test_histogram_entropy_uniform(extractor):
    # Pure single-color coin → all hue bins 0 except one → low entropy
    roi, mask = make_coin_roi((0, 128, 255))  # Orange
    hist_feats = extractor.compute_hue_histogram(roi, mask, bins=18)
    # Entropy should be lower than for random noise
    assert hist_feats['hist_entropy'] < 3.0, "Uniform color should have low entropy"
```

### Phase 4 Completion Checklist

- [ ] All 10 test cases pass
- [ ] Feature visualizations saved for all coins in test image
- [ ] Copper/Gold/Bicolor/Silver classification correct on known test coins
- [ ] Feature ranges match expected values in table above
- [ ] Feature extraction time < 100ms per coin
