# 🗂️ PROJECT STRUCTURE
## Coin Counting System — File Tree & Descriptions

```
coin_counter/
│
├── main.py                         # CLI entry point; orchestrates the full pipeline
├── requirements.txt                # All Python dependencies (pip install -r)
├── PRD.md                          # Product Requirements Document
├── PROJECT_STRUCTURE.md            # This file — annotated directory tree
├── config.yaml                     # Tunable parameters (CHT settings, thresholds)
│
├── src/                            # Core source modules (one per phase)
│   ├── __init__.py                 # Package marker; exports phase classes
│   ├── phase1_preprocessing.py     # Image loading, grayscale, CLAHE, blur, Canny
│   ├── phase2_detection.py         # Circular Hough Transform, NMS, circle filtering
│   ├── phase3_segmentation.py      # ROI extraction, circular masking, normalization
│   ├── phase4_features.py          # Size ratio, HSV/LAB stats, histogram, dominant color
│   ├── phase5_classification.py    # Rule-based classifier, confidence, counting, totaling
│   ├── phase6_output.py            # Annotated image, JSON export, HTML report, table
│   └── utils/
│       ├── __init__.py
│       ├── image_io.py             # load_image(), save_image(), validate_path()
│       ├── geometry.py             # circle_iou(), nms_circles(), crop_circle()
│       ├── color_utils.py          # bgr_to_hsv_mean(), bgr_to_lab_mean(), dom_color()
│       └── logger.py              # Loguru setup; per-run log file configuration
│
├── docs/                           # Phase planning documents
│   ├── phase1_details.md           # Phase 1: Setup & Preprocessing
│   ├── phase2_details.md           # Phase 2: CHT Detection
│   ├── phase3_details.md           # Phase 3: Segmentation
│   ├── phase4_details.md           # Phase 4: Feature Extraction
│   ├── phase5_details.md           # Phase 5: Classification & Counting
│   └── phase6_details.md           # Phase 6: Output & Visualization
│
├── data/
│   ├── sample_images/              # Raw test images (.jpg / .png)
│   │   ├── test_single_coin.jpg    # One coin, white background
│   │   ├── test_mixed_coins.jpg    # Multiple denominations
│   │   ├── test_overlapping.jpg    # Partially overlapping coins
│   │   ├── test_complex_bg.jpg     # Coins on textured surface
│   │   └── test_low_light.jpg      # Underexposed image
│   └── ground_truth/
│       ├── annotations.json        # {filename: [{x,y,r,denomination,value}]}
│       └── README.md               # How ground truth was produced
│
├── outputs/                        # All generated results (git-ignored)
│   ├── annotated/                  # Annotated result images
│   ├── intermediate/               # Per-phase debug images
│   │   ├── phase1/                 # Preprocessed frames
│   │   ├── phase2/                 # Circle overlay images
│   │   ├── phase3/                 # Individual coin ROI crops
│   │   └── phase4/                 # Feature visualization plots
│   ├── reports/                    # JSON + HTML reports per run
│   └── logs/                       # Per-run log files
│
└── tests/                          # pytest test suite
    ├── __init__.py
    ├── conftest.py                  # Shared fixtures (sample images, expected results)
    ├── test_phase1_preprocessing.py # Unit tests for preprocessing functions
    ├── test_phase2_detection.py     # CHT detection accuracy tests
    ├── test_phase3_segmentation.py  # ROI extraction & mask tests
    ├── test_phase4_features.py      # Feature vector shape & value range tests
    ├── test_phase5_classification.py# Classification accuracy & counting tests
    ├── test_phase6_output.py        # File creation & JSON schema tests
    └── test_integration.py          # Full pipeline end-to-end tests
```

---

## Key File Descriptions

### `main.py`
The CLI entry point. Parses arguments (`--image`, `--output-dir`, `--debug`, `--config`), instantiates each phase class in order, passes outputs between phases, and writes the final result. Run with:
```bash
python main.py --image data/sample_images/test_mixed_coins.jpg --debug
```

### `config.yaml`
Central configuration file. Contains:
- CHT parameters (`dp`, `minDist`, `param1`, `param2`, `minRadius`, `maxRadius`)
- Preprocessing parameters (blur kernel size, CLAHE clip limit)
- Classification thresholds (color ranges, size bins)
- Output paths and report format flags

### `src/phase1_preprocessing.py`
**Class**: `Preprocessor`  
**Key methods**: `load()`, `to_grayscale()`, `apply_clahe()`, `gaussian_blur()`, `canny_edges()`, `run()`  
Saves all intermediate images when `debug=True`.

### `src/phase2_detection.py`
**Class**: `CoinDetector`  
**Key methods**: `detect_circles()`, `filter_circles()`, `non_max_suppression()`, `run()`  
Returns list of `(x, y, r)` tuples for all detected coins.

### `src/phase3_segmentation.py`
**Class**: `CoinSegmenter`  
**Key methods**: `crop_roi()`, `apply_circular_mask()`, `normalize_roi()`, `run()`  
Returns list of normalized 128×128 BGR coin images.

### `src/phase4_features.py`
**Class**: `FeatureExtractor`  
**Key methods**: `compute_size_feature()`, `compute_color_features()`, `compute_histogram()`, `build_feature_vector()`, `run()`  
Returns `List[Dict]` — one feature dict per coin.

### `src/phase5_classification.py`
**Class**: `CoinClassifier`  
**Key methods**: `classify_coin()`, `assign_confidence()`, `count_coins()`, `total_value()`, `run()`  
Returns classification results with denomination, value, confidence, and totals.

### `src/phase6_output.py`
**Class**: `ResultExporter`  
**Key methods**: `draw_annotations()`, `save_annotated_image()`, `export_json()`, `print_table()`, `generate_html_report()`, `run()`

### `src/utils/geometry.py`
Helper functions: `circle_iou(c1, c2)` (IoU for two circles), `nms_circles(circles, iou_thresh)`, `crop_circle(image, x, y, r, padding)`.

### `src/utils/color_utils.py`
Helper functions: `bgr_to_hsv_mean(roi)`, `bgr_to_lab_mean(roi)`, `dominant_color_kmeans_free(roi, n=3)` (uses histogram clustering, no sklearn).

### `tests/conftest.py`
Pytest fixtures: `synthetic_white_circle_image()`, `sample_preprocessed_image()`, `mock_circles()`, `ground_truth_annotations()`.

### `data/ground_truth/annotations.json`
Schema:
```json
{
  "test_mixed_coins.jpg": {
    "total_value_eur": 3.88,
    "coins": [
      { "x": 120, "y": 145, "r": 48, "denomination": "50c", "value": 0.50 },
      ...
    ]
  }
}
```
