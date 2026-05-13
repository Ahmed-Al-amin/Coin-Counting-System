# Complete Project Plan: Generic Coin Counting System
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
The Generic Coin Counting System is a computer vision application that automatically detects, validates, and categorizes coins from a single photograph using only classical image processing techniques. The system is designed to be currency-agnostic, focusing on counting coins and grouping them by size (Small, Medium, Large).

### 1.2 Educational Objectives
| Objective | How It's Achieved |
| :--- | :--- |
| Understand Hough Transform | Implement CHT parameter tuning for circle detection |
| Master image preprocessing | Apply CLAHE, Gaussian blur, and Canny edge detection |
| Learn feature engineering | Extract size ratios and HSV/LAB color statistics |
| Build modular pipelines | Connect 6 independent phases with clear interfaces |
| Implement heuristic validation | Use multi-feature checks to filter noise from coins |

### 1.3 Success Criteria
*   **Detection Rate:** ≥ 90% of coins correctly located
*   **Categorization Accuracy:** ≥ 90% of coins correctly categorized by size
*   **Total Count Error:** 0 (Exact match)
*   **Processing Time:** < 5 seconds for 1080p image on standard CPU
*   **Zero ML/DL:** No TensorFlow, PyTorch, or training code

### 1.4 Size Categorization Table
| Category | Rel. Radius (r/diag) | Notes |
| :--- | :--- | :--- |
| Small | < 0.035 | e.g., 1c, 2c, 10c EUR |
| Medium | 0.035 - 0.055 | e.g., 5c, 20c, 50c, €1 EUR |
| Large | > 0.055 | e.g., €2 EUR, $1 USD |

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
│  │ Load Image   │→ │ Grayscale    │→ │ CLAHE        │→ │ Bilateral    │    │
│  │ Validate     │  │ Conversion   │  │ Contrast     │  │ Filter       │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                           │                                      │          │
│                           └──────────┬───────────────────────────┘          │
│                                      ▼                                      │
│                           ┌──────────────────────┐                          │
│                           │   Sobel Magnitude    │                          │
│                           └──────────────────────┘                          │
│  OUTPUT: Gradient map + scaled original                                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 2: HYBRID DETECTION & SHAPE CLOSING                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  1. Primary: cv2.HoughCircles(bilateral, ...)                        │  │
│  │  2. Closing: Morphological Closing (Bridge broken edges)             │  │
│  │  3. Helper:  Contour analysis (Area + Circularity + Radius check)    │  │
│  │  4. Refinement: Centroid Clustering + One-Circle-Per-Coin Enforcement│  │
│  └──────────────────────────────────────────────────────────────────────┘  │
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
│  │  Size: rel_size = r / diagonal                                      │    │
│  │  Color: Mean HSV, Mean LAB, Hue Histogram Entropy                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  OUTPUT: Feature vector dict per candidate circle                           │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PHASE 5: VALIDATION & CATEGORIZATION                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  1. Check if candidate is coin (based on color, entropy, saturation)│   │
│  │  2. Assign size category (Small, Medium, Large)                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OUTPUT: Validated coins + total count + breakdown                         │
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

---

## 3. PHASE-BY-PHASE EXECUTION PLAN
(Phases 1-4 remain largely identical to previous version)

### Phase 5: Validation & Categorization
**Duration:** 6 hours | **Priority:** HIGH

#### Implementation Steps
| Step | Action | Code Location | Expected Output |
| :--- | :--- | :--- | :--- |
| 5.1 | Implement `_validate_coin` heuristic | `src/classification.py` | is_coin boolean |
| 5.2 | Implement size-based categorization | `src/classification.py` | Category label |
| 5.3 | Add confidence scoring | `src/classification.py` | 0-1 confidence |
| 5.4 | Implement counting and summarizing | `src/classification.py` | Category breakdown |

---

## 4. COMPLETE INTEGRATION PLAN
### 4.1 main.py Implementation
Updated to report `total_count` and generic categories.

### 4.2 config.yaml Configuration
```yaml
classification:
  size_thresholds:
    small_medium: 0.035
    medium_large: 0.055
  low_confidence_threshold: 0.50
```

---

## 5. END-TO-END TEST SUITE
Updated tests in `tests/test_phase5_classification.py` to assert counts and categories.

---
