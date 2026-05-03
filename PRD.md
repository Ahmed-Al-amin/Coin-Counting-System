# 📄 Product Requirements Document (PRD)
## Coin Counting System — Classical Computer Vision

---

## 1. Executive Summary

The **Coin Counting System** is a classical computer vision pipeline that automatically detects, classifies, and totals the monetary value of coins visible in a single photograph. The system uses **no machine learning or deep learning** — all classification is performed through geometric and colorimetric analysis using the Circular Hough Transform (CHT) and hand-crafted features.

---

## 2. Background & Motivation

| Problem | Solution |
|---|---|
| Manual coin counting is slow and error-prone | Automated image-based recognition |
| ML models require labeled training data | Pure classical CV — no training needed |
| Embedded/offline deployments need lightweight code | NumPy + OpenCV only, no GPU |
| CV course assignments need explainable pipelines | Each stage is deterministic and inspectable |

---

## 3. Scope

### 3.1 In Scope
- Detection of **Euro coins** (1c, 2c, 5c, 10c, 20c, 50c, €1, €2)
- Single flat-surface images (desk, table, white background)
- Support for **overlapping coins** (partial)
- Detection of **multiple denominations** in one image
- Output: annotated image + total value + per-denomination count

### 3.2 Out of Scope
- Video or real-time webcam streams
- Stacked or 3D coin arrangements
- Non-Euro currencies (extensible but not implemented)
- Worn, heavily corroded, or damaged coins
- Deep learning or neural network approaches

---

## 4. Functional Requirements

### FR-01: Image Ingestion
- Accept JPEG, PNG, TIFF, BMP images
- Support resolution from 480p to 4K
- Validate image readability before processing

### FR-02: Preprocessing
- Convert to grayscale for CHT
- Apply Gaussian blur to reduce noise
- Normalize exposure/contrast (CLAHE)
- Output preprocessed image for inspection

### FR-03: Coin Detection
- Locate all circular regions using Circular Hough Transform
- Return center (x, y) and radius r for each detected circle
- Minimum coin diameter: 15px; Maximum: 500px
- False-positive rate < 10% on benchmark set

### FR-04: Segmentation
- Extract each detected coin as a cropped circular ROI
- Mask out background within each ROI
- Normalize ROI to standard 128×128 px canvas

### FR-05: Feature Extraction
- **Size feature**: normalized radius relative to image diagonal
- **Color features**: mean HSV, mean LAB, dominant color cluster
- Combine into a feature vector per coin

### FR-06: Classification
- Assign denomination using rule-based decision tree on features
- Confidence score (0–1) per classification
- Flag low-confidence detections for review

### FR-07: Counting & Totaling
- Count coins per denomination
- Compute total monetary value in EUR
- Output per-denomination breakdown

### FR-08: Visualization & Output
- Annotated image with bounding circles, denomination labels, confidence
- Console summary table
- JSON results file
- Optional: PDF/HTML report

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | Process 1080p image in < 5 seconds on CPU |
| **Accuracy** | ≥ 85% classification accuracy on clean test images |
| **Reliability** | Graceful error handling; never crash silently |
| **Portability** | Runs on Windows, macOS, Linux (Python 3.9+) |
| **Maintainability** | Each phase in its own module; unit tested |
| **Transparency** | Save intermediate images at each pipeline stage |

---

## 6. System Architecture Overview

```
Input Image
     │
     ▼
┌─────────────────────┐
│  Phase 1            │  Grayscale, CLAHE, Gaussian Blur,
│  Preprocessing      │  Edge Detection (Canny)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 2            │  Circular Hough Transform → (x, y, r)
│  CHT Detection      │  NMS, Min/Max radius filtering
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 3            │  Crop ROI, Circular mask,
│  Segmentation       │  Normalize to 128×128
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 4            │  Radius ratio, Mean HSV/LAB,
│  Feature Extraction │  Color histogram, Dominant hue
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 5            │  Rule-based classifier,
│  Classification     │  Confidence scoring, Counting
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 6            │  Annotated image, JSON, Table,
│  Output             │  Optional HTML report
└─────────────────────┘
```

---

## 7. Euro Coin Reference Data

| Coin | Diameter (mm) | Color | Notes |
|---|---|---|---|
| 1 cent | 16.25 | Copper | Smallest |
| 2 cent | 18.75 | Copper | |
| 5 cent | 21.25 | Copper | Largest copper |
| 10 cent | 19.75 | Gold | |
| 20 cent | 22.25 | Gold | |
| 50 cent | 24.25 | Gold | Largest gold |
| 1 euro | 23.25 | Bicolor (silver+gold) | |
| 2 euro | 25.75 | Bicolor (gold+silver) | Largest |

*Color groups: Copper (#CE8946 range), Gold (#D4AF37 range), Silver (#C0C0C0 range)*

---

## 8. Acceptance Criteria

| ID | Criterion | Pass Threshold |
|---|---|---|
| AC-01 | Detect all coins in clean background image | ≥ 95% recall |
| AC-02 | Correct denomination classification | ≥ 85% accuracy |
| AC-03 | Total value computation | Exact match or ±1 cent |
| AC-04 | Processing time per image | < 5 seconds |
| AC-05 | No crash on valid input | 100% |
| AC-06 | Output annotated image saved | 100% |

---

## 9. Deliverables

1. `main.py` — CLI entry point
2. `src/` — Six phase modules
3. `tests/` — pytest test suite
4. `data/` — Sample images + ground truth
5. `outputs/` — Generated results
6. `docs/` — Phase detail markdown files
7. `requirements.txt`, `PRD.md`, `PROJECT_STRUCTURE.md`

---

## 10. Timeline

| Phase | Description | Est. Hours |
|---|---|---|
| 1 | Setup & Preprocessing | 4h |
| 2 | CHT Detection | 6h |
| 3 | Segmentation | 4h |
| 4 | Feature Extraction | 6h |
| 5 | Classification & Counting | 8h |
| 6 | Output & Visualization | 4h |
| — | Integration & Testing | 4h |
| **Total** | | **~36h** |

---

*Document version: 1.0 — Coin Counting System CV Assignment*
