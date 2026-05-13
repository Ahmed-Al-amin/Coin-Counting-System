# 📄 Product Requirements Document (PRD)
## Generic Coin Counting System — Classical Computer Vision

---

## 1. Executive Summary

The **Generic Coin Counting System** is a classical computer vision pipeline that automatically detects, validates, and categorizes coins visible in a single photograph. The system uses **no machine learning** — detection is performed via the Circular Hough Transform (CHT), and categorization is based on relative size and colorimetric analysis.

---

## 2. Background & Motivation

| Problem | Solution |
|---|---|
| Manual coin counting is slow | Automated image-based recognition |
| ML models require labeled data | Pure classical CV — no training needed |
| Multi-currency support is hard | Generic size/color categorization |
| CV course assignments | Explainable pipelines and feature engineering |

---

## 3. Scope

### 3.1 In Scope
- Detection of circular coins of any currency
- Categorization into **Small, Medium, and Large** groups
- Single flat-surface images (desk, table, white background)
- Support for **overlapping coins** (partial)
- Output: annotated image + total count + size breakdown

### 3.2 Out of Scope
- Video or real-time webcam streams
- Stacked or 3D coin arrangements
- Specific monetary value (EUR, USD, etc.) determination
- Worn, heavily corroded, or damaged coins
- Deep learning or neural network approaches

---

## 4. Functional Requirements

### FR-01: Image Ingestion
- Accept JPEG, PNG, TIFF, BMP images
- Support resolution from 480p to 4K

### FR-02: Preprocessing
- Convert to grayscale for CHT
- Apply Gaussian blur and CLAHE contrast enhancement
- Output preprocessed image for inspection

### FR-03: Coin Detection
- Locate all circular regions using Circular Hough Transform
- **Fallback Mechanism**: Implement contour-based detection for cases where CHT fails (e.g., low contrast edges).
- Return center (x, y) and radius r for each detected circle
- False-positive rate < 10% on benchmark set

### FR-04: Segmentation
- Extract each detected circle as a cropped circular ROI
- Mask out background and normalize to 128×128 px

### FR-05: Feature Extraction
- **Size feature**: normalized radius relative to image diagonal
- **Color features**: mean HSV, mean LAB, hue histogram entropy
- Combine into a feature vector per candidate

### FR-06: Validation & Categorization
- Validate candidates as "Coin" or "Noise" using feature heuristics
- Assign size category (Small/Medium/Large) using relative radius
- Confidence score (0–1) per detection

### FR-07: Counting & Summary
- Count total validated coins
- Compute breakdown per size category
- Output summary statistics

### FR-08: Visualization & Output
- Annotated image with bounding circles, size labels, confidence
- Console summary table
- JSON results file and HTML report

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| **Performance** | Process 1080p image in < 5 seconds on CPU |
| **Accuracy** | ≥ 90% detection rate on clean test images |
| **Reliability** | Graceful error handling; never crash silently |
| **Maintainability** | Modular pipeline; unit tested |

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
│  Phase 2            │  CHT + Contour Fallback,
│  Detection          │  Clustering, Radius Consistency
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
│  Feature Extraction │  Hue histogram entropy
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 5            │  Generic validator,
│  Categorization     │  Size-based grouping
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 6            │  Annotated image, JSON, Table,
│  Output             │  HTML report
└─────────────────────┘
```

---

## 7. Categorization Reference

| Category | Relative Size (Radius / Diagonal) |
|---|---|
| Small | < 0.035 |
| Medium | 0.035 – 0.055 |
| Large | > 0.055 |

---

## 8. Acceptance Criteria

| ID | Criterion | Pass Threshold |
|---|---|---|
| AC-01 | Detect coins in clean background image | ≥ 95% recall |
| AC-02 | Correct size categorization | ≥ 90% accuracy |
| AC-03 | Total count accuracy | Exact match |
| AC-04 | Processing time per image | < 5 seconds |
| AC-05 | No crash on valid input | 100% |
| AC-06 | Output annotated image saved | 100% |

---

## 9. Deliverables

1. `main.py` — CLI entry point
2. `src/` — Six phase modules
3. `tests/` — pytest test suite
4. `data/` — Sample images
5. `outputs/` — Generated results
6. `docs/` — Phase detail markdown files

---

*Document version: 2.0 — Generic Coin Counting System*
