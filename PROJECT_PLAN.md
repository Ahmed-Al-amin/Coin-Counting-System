# Complete Project Plan: Coin Counting System
## Classical Computer Vision Implementation

## 📋 TABLE OF CONTENTS
1. [Project Overview & Purpose](#1-project-overview--purpose)
2. [System Architecture](#2-system-architecture)
3. [Phase-by-Phase Execution Plan](#3-phase-by-phase-execution-plan)
4. [Complete Integration Plan](#4-complete-integration-plan)
5. [End-to-End Test Suite](#5-end-to-end-test-suite)
6. [Deployment & Usage Guide](#6-deployment--usage-guide)
7. [Troubleshooting & Optimization](#7-troubleshooting--optimization)

---

## 1. PROJECT OVERVIEW & PURPOSE
### 1.1 Project Description
The Coin Counting System is a computer vision application that automatically detects, classifies, and counts coins from a single photograph using only classical image processing techniques. No machine learning or deep learning models are used — all detection relies on the Circular Hough Transform (CHT) for geometry and hand-crafted color features for classification.

### 1.2 Educational Objectives
| Objective | How It's Achieved |
| :--- | :--- |
| Understand Hough Transform | Implement CHT parameter tuning for circle detection |
| Master image preprocessing | Apply CLAHE, Gaussian blur, and Canny edge detection |
| Learn feature engineering | Extract size ratios and HSV/LAB color statistics |
| Build modular pipelines | Connect 6 independent phases with clear interfaces |
| Test computer vision systems | Write comprehensive test cases for each stage |

### 1.3 Success Criteria
*   **Detection Rate:** ≥ 90% of coins correctly located
*   **Classification Accuracy:** ≥ 85% of coins correctly denominated
*   **Total Value Error:** ≤ 2 cents per image
*   **Processing Time:** < 5 seconds for 1080p image on standard CPU
*   **Zero ML/DL:** No TensorFlow, PyTorch, or training code

### 1.4 Euro Coin Reference Table
| Coin | Diameter (mm) | Color Group | Value (€) |
| :--- | :--- | :--- | :--- |
| 1 cent | 16.25 | Copper | 0.01 |
| 2 cents | 18.75 | Copper | 0.02 |
| 5 cents | 21.25 | Copper | 0.05 |
| 10 cents | 19.75 | Gold | 0.10 |
| 20 cents | 22.25 | Gold | 0.20 |
| 50 cents | 24.25 | Gold | 0.50 |
| 1 euro | 23.25 | Bicolor | 1.00 |
| 2 euros | 25.75 | Bicolor | 2.00 |

---

## 2. SYSTEM ARCHITECTURE
### 2.1 Pipeline Flow Diagram
```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Image File (JPG/PNG)                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: PREPROCESSING                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Load Image   │→ │ Grayscale    │→ │ CLAHE        │→ │ Gaussian     │    │
│  │ Validate     │  │ Conversion   │  │ Contrast     │  │ Blur         │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                           │                                      │          │
│                           └──────────┬───────────────────────────┘          │
│                                      ▼                                      │
│                           ┌──────────────────────┐                          │
│                           │ Canny Edge Detection │                          │
│                           └──────────────────────┘                          │
│  OUTPUT: Edge map (binary image) + scaled original                          │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: CIRCULAR HOUGH TRANSFORM DETECTION                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  cv2.HoughCircles(edges, HOUGH_GRADIENT, dp, minDist, param1, param2)│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                    ┌─────────────────┼─────────────────┐                   │
│                    ▼                 ▼                 ▼                   │
│            ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │
│            │ NMS         │   │ Boundary    │   │ Radius      │             │
│            │ Filter      │   │ Filter      │   │ Validation  │             │
│            └─────────────┘   └─────────────┘   └─────────────┘             │
│                                                                             │
│  OUTPUT: List of (x, y, r) circles                                         │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 3: SEGMENTATION                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ ROI Crop     │→ │ Circular     │→ │ Mask         │→ │ Normalize    │    │
│  │ (with padding)│  │ Mask Create  │  │ Application  │  │ to 128×128   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
│  OUTPUT: List of normalized 128×128 coin images (BGR, mask = black bg)     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 4: FEATURE EXTRACTION                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  For each normalized coin:                                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │ Size:        │  │ HSV:         │  │ LAB:         │              │    │
│  │  │ rel_size =   │  │ hue_mean,    │  │ L_mean,      │              │    │
│  │  │ r / diagonal │  │ sat_mean,    │  │ a_mean,      │              │    │
│  │  │              │  │ val_mean     │  │ b_mean       │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │                                                                     │    │
│  │  ┌──────────────────────────────────────────────────────────────┐  │    │
│  │  │ Hue Histogram: 18 bins, normalized, dominant_hue, entropy    │  │    │
│  │  └──────────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  OUTPUT: Feature vector dict per coin                                       │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: CLASSIFICATION & COUNTING                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TWO-STAGE DECISION TREE                          │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ Color Group     │                                                │   │
│  │  │ Classification  │                                                │   │
│  │  └────────┬────────┘                                                │   │
│  │     ┌─────┼─────┬─────────┐                                         │   │
│  │     ▼     ▼     ▼         ▼                                         │   │
│  │  COPPER  GOLD  BICOLOR  SILVER                                      │   │
│  │     │     │     │         │                                         │   │
│  │     ▼     ▼     ▼         ▼                                         │   │
│  │  Size   Size   Size    UNKNOWN                                      │   │
│  │   ↓      ↓      ↓                                                  │   │
│  │ 1c/2c/5c 10c/20c/50c  €1/€2                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: Classified coins + total value + breakdown                        │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 6: OUTPUT & VISUALIZATION                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Annotated    │  │ JSON Export  │  │ Console      │  │ HTML Report  │    │
│  │ Image with   │  │ Machine-     │  │ Table        │  │ Visual       │    │
│  │ Circles &    │  │ Readable     │  │ Formatted    │  │ Summary      │    │
│  │ Labels       │  │              │  │              │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
│  OUTPUT: annotated.png, results.json, console table, report.html           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Between Phases
```python
# Expected output structure from each phase
phase1_output = {
    'original': np.ndarray,      # Original loaded image
    'scaled': np.ndarray,        # Resized to working resolution
    'gray': np.ndarray,          # Grayscale version
    'clahe': np.ndarray,         # Contrast enhanced
    'blurred': np.ndarray,       # Gaussian blurred
    'edges': np.ndarray,         # Canny edge map
    'scale_factor': float        # Original→scaled ratio
}

phase2_output = {
    'circles': List[(x, y, r)],  # Detected circles
    'count': int,                # Number of circles
    'visualization': np.ndarray, # Debug image with circles drawn
    'scale_factor': float        # Passed through
}

phase3_output = {
    'segments': List[Dict],      # Each with: norm_roi, masked_roi, circle, etc.
    'count': int
}

phase4_output = {
    'features': List[Dict],      # Feature vectors for each coin
    'image_diagonal': float
}

phase5_output = {
    'classified_coins': List[Dict],  # With denomination, confidence
    'summary': Dict                  # total_eur, breakdown, counts
}

phase6_output = {
    'annotated_image': np.ndarray,
    'annotated_path': str,
    'json_path': str,
    'html_path': str
}
```

---

## 3. PHASE-BY-PHASE EXECUTION PLAN
### Phase 1: Image Preprocessing
**Duration:** 4 hours | **Priority:** HIGH

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 1.1 | Create project structure | Terminal | Directory tree |
| 1.2 | Install OpenCV, NumPy, Matplotlib | `pip install -r requirements.txt` | Dependencies ready |
| 1.3 | Implement image loading with validation | `src/utils/image_io.py` | `load_image()` function |
| 1.4 | Add grayscale conversion | `phase1_preprocessing.py` | Gray image |
| 1.5 | Implement CLAHE contrast enhancement | `phase1_preprocessing.py` | Enhanced image |
| 1.6 | Add Gaussian blur | `phase1_preprocessing.py` | Smoothed image |
| 1.7 | Implement Canny with auto-thresholds | `phase1_preprocessing.py` | Edge map |
| 1.8 | Add scale normalization | `phase1_preprocessing.py` | Resized image |

#### Critical Parameters
```yaml
preprocessing:
  max_working_dim: 1024      # Resize longest side to 1024px
  clahe_clip_limit: 2.0      # Contrast enhancement strength
  clahe_tile_size: [8, 8]    # Local histogram tiles
  blur_kernel: [5, 5]        # Must be odd numbers
  blur_sigma: 1.5            # Gaussian standard deviation
```

---

### Phase 2: Circular Hough Transform Detection
**Duration:** 6 hours | **Priority:** HIGH

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 2.1 | Implement `cv2.HoughCircles` wrapper | `phase2_detection.py` | Raw circles |
| 2.2 | Add radius range auto-estimation | `phase2_detection.py` | `min_r`, `max_r` |
| 2.3 | Implement NMS for overlapping circles | `phase2_detection.py` | Deduplicated list |
| 2.4 | Add boundary filtering | `phase2_detection.py` | Edge circles removed |
| 2.5 | Create detection visualization | `phase2_detection.py` | Green circles drawn |
| 2.6 | Add multi-scale detection (optional) | `phase2_detection.py` | Better coverage |

---

### Phase 3: Coin Segmentation
**Duration:** 4 hours | **Priority:** MEDIUM

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 3.1 | Implement ROI cropping with padding | `phase3_segmentation.py` | Square crop per coin |
| 3.2 | Create circular mask generation | `phase3_segmentation.py` | Binary circle mask |
| 3.3 | Apply mask to ROI | `phase3_segmentation.py` | Background = black |
| 3.4 | Normalize to 128×128 | `phase3_segmentation.py` | Uniform size |
| 3.5 | Batch process all coins | `phase3_segmentation.py` | List of segments |
| 3.6 | Create debug grid visualization | `phase3_segmentation.py` | All coins in one image |

---

### Phase 4: Feature Extraction
**Duration:** 6 hours | **Priority:** HIGH

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 4.1 | Compute size feature (`rel_size`) | `phase4_features.py` | Normalized radius |
| 4.2 | Convert ROI to HSV | `phase4_features.py` | HSV image |
| 4.3 | Compute masked HSV statistics | `phase4_features.py` | `hue_mean`, `sat_mean` |
| 4.4 | Convert ROI to LAB | `phase4_features.py` | LAB image |
| 4.5 | Compute masked LAB statistics | `phase4_features.py` | `L_mean`, `a_mean`, `b_mean` |
| 4.6 | Build hue histogram (18 bins) | `phase4_features.py` | Normalized histogram |
| 4.7 | Determine dominant hue | `phase4_features.py` | Peak bin center |
| 4.8 | Classify color group | `phase4_features.py` | String label |

---

### Phase 5: Classification & Counting
**Duration:** 8 hours | **Priority:** HIGH

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 5.1 | Create denomination lookup table | `phase5_classification.py` | `EURO_COINS` dict |
| 5.2 | Implement copper group classifier | `phase5_classification.py` | Denomination |
| 5.3 | Implement gold group classifier | `phase5_classification.py` | Denomination |
| 5.4 | Implement bicolor classifier | `phase5_classification.py` | Denomination |
| 5.5 | Add confidence scoring | `phase5_classification.py` | 0-1 confidence |
| 5.6 | Implement counting and totaling | `phase5_classification.py` | Breakdown |
| 5.7 | Add cross-coin consistency check | `phase5_classification.py` | Adjusted confidence |

---

### Phase 6: Output & Visualization
**Duration:** 4 hours | **Priority:** MEDIUM

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 6.1 | Draw colored circles on image | `phase6_output.py` | Circle per coin |
| 6.2 | Add denomination labels above coins | `phase6_output.py` | Text labels |
| 6.3 | Add confidence bars below coins | `phase6_output.py` | Visual confidence |
| 6.4 | Create summary banner at top | `phase6_output.py` | Total + count |
| 6.5 | Save annotated image | `phase6_output.py` | PNG file |
| 6.6 | Export JSON results | `phase6_output.py` | JSON file |
| 6.7 | Print console table | `phase6_output.py` | Formatted table |
| 6.8 | Generate HTML report | `phase6_output.py` | Self-contained HTML |

---

## 4. COMPLETE INTEGRATION PLAN
### 4.1 main.py Implementation
```python
#!/usr/bin/env python3
"""
Coin Counting System - Main Entry Point
Usage: python main.py --image path/to/coins.jpg [--debug] [--config config.yaml]
"""

import argparse
import yaml
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.phase1_preprocessing import Preprocessor
from src.phase2_detection import CoinDetector
from src.phase3_segmentation import CoinSegmenter
from src.phase4_features import FeatureExtractor
from src.phase5_classification import CoinClassifier
from src.phase6_output import ResultExporter

# ... (rest of main.py implementation)
```

### 4.2 config.yaml Complete Configuration
```yaml
# Coin Counting System Configuration
# Adjust these parameters based on your camera setup

preprocessing:
  max_working_dim: 1024
  clahe_clip_limit: 2.0
  clahe_tile_size: [8, 8]
  blur_kernel: [5, 5]
  blur_sigma: 1.5
  canny_low_ratio: 0.5
  canny_high_ratio: 1.0

detection:
  dp: 1.2
  min_dist: 30
  param1: 80
  param2: 30
  min_radius: 15
  max_radius: 200
  nms_iou_threshold: 0.3
  boundary_margin: 0.5

segmentation:
  target_size: 128
  padding_ratio: 0.05
  mask_erosion_px: 2

features:
  size_thresholds:
    tiny: 0.025
    small: 0.040
    medium: 0.055
    large: 0.070
  bicolor_hue_std_thresh: 22
  hue_hist_bins: 18

classification:
  size_splits:
    copper_1c_2c: 0.022
    copper_2c_5c: 0.032
    gold_10c_20c: 0.030
    gold_20c_50c: 0.038
    bicolor_1e_2e: 0.043
  low_confidence_threshold: 0.50

output:
  output_dir: outputs
  generate_html: true
  annotate_confidence_bar: true
  banner_height_px: 60
```

---

## 5. END-TO-END TEST SUITE
### 5.1 Test File: tests/test_integration.py
The full integration test suite is located in `tests/test_integration.py`. This file contains 10+ test cases covering:
*   Full pipeline on synthetic coins
*   Empty image handling
*   Touching coins detection
*   Ground truth comparison
*   Classification accuracy
*   Total value calculation
*   Output file generation
*   Performance requirements
*   Scale invariance
*   Color consistency

### 5.2 Running the Tests
```bash
# Run all tests
pytest tests/test_integration.py -v

# Run specific test
pytest tests/test_integration.py::TestIntegration::test_full_pipeline_non_touching -v

# Run with coverage report
pytest tests/test_integration.py --cov=src --cov-report=html
```

---

## 6. DEPLOYMENT & USAGE GUIDE
### 6.1 Installation
```bash
# Clone or create project
mkdir coin_counter
cd coin_counter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 6.2 Basic Usage
```bash
# Process a single image
python main.py --image data/sample_images/coins.jpg

# With debug output (saves intermediate images)
python main.py --image coins.jpg --debug
```

---

## 7. TROUBLESHOOTING & OPTIMIZATION
### 7.1 Common Issues and Solutions
| Issue | Likely Cause | Solution |
| :--- | :--- | :--- |
| No coins detected | param2 too high | Decrease to 20-25 |
| Too many false circles | param2 too low | Increase to 40-50 |
| Missed overlapping coins | minDist too high | Reduce to 15-20 |
| Wrong color classification | Lighting variation | Adjust hue thresholds |

---

## 8. FINAL DELIVERABLES CHECKLIST
*   `main.py` - Entry point with CLI
*   `src/` - Implementation files for all 6 phases
*   `requirements.txt` - Dependencies
*   `config.yaml` - Configuration parameters
*   `tests/test_integration.py` - Integration tests
*   Documentation (`PRD.md`, `PROJECT_PLAN.md`, etc.)
