# Phase 2: CHT Detection & Fallback
## Generic Coin Counting System — Classical CV Pipeline

---

## 1. Overview

Detection is the process of locating potential coins in the preprocessed image. The system primarily uses the **Circular Hough Transform (CHT)** for its robustness to noise and partial occlusions. However, for cases where CHT may fail (e.g., extremely low contrast or non-perfectly circular edges), a **Contour Detection Fallback** is implemented.

---

## 2. Primary Detection: Circular Hough Transform (CHT)

CHT detects circles by voting in a 3D parameter space (center_x, center_y, radius).

### 2.1 Mathematical Foundation

A circle is defined by:
```
(x - a)² + (y - b)² = r²
```
where `(a, b)` is the center and `r` is the radius.

### 2.2 OpenCV Implementation

We use `cv2.HoughCircles` with the `HOUGH_GRADIENT` method. This is more memory-efficient as it uses the gradient of the edges to reduce the voting space.

---

## 3. Guided Hybrid Detection Logic

The system now utilizes a **Contours-First** guided detection strategy. Instead of running both detectors independently, the system uses initial shape analysis to inform the Circular Hough Transform.

### 3.1 Step 1: Initial Contour Analysis
Contours are extracted from the morphologically closed edge map. These provide a preliminary list of candidates and, more importantly, a statistical profile of the "typical" object size in the scene.

### 3.2 Step 2: Robust Radius Estimation
The system analyzes the distribution of radii from the initial contours. It uses a **percentile-based filter (20th–80th)** to reject outliers and calculates a robust median radius. A dynamic search range ($[0.6r, 1.5r]$) is then established.

### 3.3 Step 3: Guided Circular Hough Transform (CHT)
The Circular Hough Transform is executed on the bilateral filtered image, but its search parameters (`minRadius`, `maxRadius`) are **dynamically updated** based on the estimation from Step 2. This drastically reduces the voting space, improving both speed and precision by ignoring out-of-scale artifacts.

---

## 4. Post-Processing & Validation

All hybrid candidates undergo a rigorous refinement pipeline:
1. **NMS (0.6)**: Deduplicate overlapping detections.
2. **Scale-Aware Clustering**: Group circles by center (dist < 0.8r) and radius similarity.
3. **Edge Alignment Selection**: Within each cluster, select the circle with the strongest sum of gradient pixels along its perimeter.
4. **Position Validation**: Verify the ROI contains at least **30 edge pixels** to reject background noise.
5. **Boundary Filtering**: Remove detections too close to the image edge.

---

## 5. Optimization & Robustness (Fixes Applied)

To improve detection accuracy and reduce false positives, several optimizations have been implemented:

### 5.1 Parameter Tuning
- **param1 (120)**: Increased sensitivity for internal edge detection.
- **param2 (60)**: Increased accumulator threshold to significantly reduce noise and fake circles.
- **minDist (80)**: Prevent multiple detections for the same coin by enforcing a minimum distance between centers.

### 5.2 Radius Range Constraints
The system now auto-estimates radius range using tighter bounds relative to the image diagonal:
- **Min Radius**: 4% of diagonal
- **Max Radius**: 15% of diagonal
This eliminates tiny artifacts and massive nonsense circles.

### 5.3 Strong Non-Maximum Suppression (NMS)
- **NMS Threshold (0.6)**: Stricter overlap control to ensure only the most prominent circle is kept for each coin.
### 5.4 Scale-Aware Dynamic Clustering
The system now groups circles that belong to the same coin using a clustering algorithm with a **dynamic distance threshold** set to 80% of the candidate radius. This ensures robust grouping regardless of coin size. For each cluster, the circle with the **strongest edge alignment score** is selected.

### 5.5 Outer Boundary Validation
To reject false positives (such as circles detected on internal coin designs), a new heuristic check compares edge density in an inner core vs the outer perimeter ring. Only circles where the outer edge strength exceeds the inner design noise are preserved.

### 5.6 Edge-Based Scoring
Within each cluster, the system scores candidates based on the sum of edge pixels along their exact mathematical perimeter. This "picks" the circle that perfectly fits the physical boundary.
...

The system now processes the **Sobel magnitude map** directly instead of the blurred grayscale image. This focuses the transform on structural boundaries, drastically reducing voting noise from surface textures.

---

## 6. `config.yaml` Parameters for Phase 2

```yaml
detection:
  dp: 1.2                # Accumulator resolution ratio
  min_dist: 80           # Minimum center-to-center distance (px)
  param1: 120            # Internal Canny high threshold
  param2: 90             # Accumulator vote threshold
  min_radius: 15         # Minimum coin radius (px)
  max_radius: 200        # Maximum coin radius (px)
  nms_iou_threshold: 0.6 # Overlap threshold for NMS
  boundary_margin: 0.5   # Fraction of radius to use as boundary margin
```
