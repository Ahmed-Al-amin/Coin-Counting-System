#!/bin/bash

# ============================================================================
# Coin Counting System - Project Setup Script
# Creates complete directory structure and all required files
# ============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory (default: coin_counter)
PROJECT_NAME="${1:-coin_counter}"
PROJECT_ROOT="$(pwd)/$PROJECT_NAME"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     COIN COUNTING SYSTEM - PROJECT SETUP                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Creating project at: ${PROJECT_ROOT}${NC}"
echo ""

# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================
echo -e "${GREEN}[1/6] Creating directory structure...${NC}"

mkdir -p "$PROJECT_ROOT"/{src/utils,data/{sample_images,ground_truth},outputs/{annotated,intermediate/{phase1,phase2,phase3,phase4},reports,logs},tests,docs}

echo -e "${GREEN}✓ Directories created${NC}"

# ============================================================================
# CREATE REQUIREMENTS.TXT
# ============================================================================
echo -e "${GREEN}[2/6] Creating requirements.txt...${NC}"

cat > "$PROJECT_ROOT/requirements.txt" << 'EOF'
# Coin Counting System - Python Dependencies
# Install with: pip install -r requirements.txt

# Core computer vision
opencv-python==4.9.0.80

# Numerical computing
numpy==1.26.0

# Visualization and plotting
matplotlib==3.8.0

# Scientific computing (for potential advanced features)
scipy==1.11.0

# YAML configuration parsing
pyyaml==6.0.1

# Testing framework
pytest==7.4.3
pytest-cov==4.1.0

# Logging (optional, for better debug output)
loguru==0.7.2

# Image processing utilities
pillow==10.1.0
EOF

echo -e "${GREEN}✓ requirements.txt created${NC}"

# ============================================================================
# CREATE CONFIG.YAML
# ============================================================================
echo -e "${GREEN}[3/6] Creating config.yaml...${NC}"

cat > "$PROJECT_ROOT/config.yaml" << 'EOF'
# ============================================================================
# Coin Counting System - Configuration File
# Adjust these parameters based on your camera setup and lighting conditions
# ============================================================================

# ----------------------------------------------------------------------------
# Phase 1: Image Preprocessing
# ----------------------------------------------------------------------------
preprocessing:
  max_working_dim: 1024          # Resize longest side to this many pixels
  clahe_clip_limit: 2.0          # Contrast enhancement strength (1.0-4.0)
  clahe_tile_size: [8, 8]        # Grid size for local histogram equalization
  blur_kernel: [5, 5]            # Gaussian blur kernel (must be odd numbers)
  blur_sigma: 1.5                # Gaussian standard deviation
  canny_low_ratio: 0.5           # Canny low = Otsu * this ratio
  canny_high_ratio: 1.0          # Canny high = Otsu * this ratio

# ----------------------------------------------------------------------------
# Phase 2: Circular Hough Transform Detection
# ----------------------------------------------------------------------------
detection:
  dp: 1.2                        # Accumulator resolution ratio (1.0-2.0)
  min_dist: 30                   # Minimum center-to-center distance (pixels)
  param1: 80                     # Internal Canny high threshold
  param2: 30                     # Accumulator threshold (lower = more circles)
  min_radius: 15                 # Minimum coin radius (pixels)
  max_radius: 200                # Maximum coin radius (pixels)
  nms_iou_threshold: 0.3         # Overlap threshold for non-maximum suppression
  boundary_margin: 0.5           # Fraction of radius for boundary filtering

# ----------------------------------------------------------------------------
# Phase 3: Coin Segmentation
# ----------------------------------------------------------------------------
segmentation:
  target_size: 128               # Normalized coin image size (pixels)
  padding_ratio: 0.05            # ROI padding as fraction of radius
  mask_erosion_px: 2             # Erode circular mask by this many pixels

# ----------------------------------------------------------------------------
# Phase 4: Feature Extraction
# ----------------------------------------------------------------------------
features:
  size_thresholds:
    tiny: 0.025                  # < 2.5% of image diagonal
    small: 0.040                 # 2.5-4.0%
    medium: 0.055                # 4.0-5.5%
    large: 0.070                 # 5.5-7.0%
  bicolor_hue_std_thresh: 22     # Hue std above this = bicolor coin
  hue_hist_bins: 18              # Number of bins for hue histogram

# ----------------------------------------------------------------------------
# Phase 5: Classification & Counting
# ----------------------------------------------------------------------------
classification:
  size_splits:
    copper_1c_2c: 0.022          # rel_size threshold between 1c and 2c
    copper_2c_5c: 0.032          # rel_size threshold between 2c and 5c
    gold_10c_20c: 0.030          # rel_size threshold between 10c and 20c
    gold_20c_50c: 0.038          # rel_size threshold between 20c and 50c
    bicolor_1e_2e: 0.043         # rel_size threshold between €1 and €2
  low_confidence_threshold: 0.50 # Flag coins below this confidence

# ----------------------------------------------------------------------------
# Phase 6: Output & Visualization
# ----------------------------------------------------------------------------
output:
  output_dir: outputs
  generate_html: true            # Generate HTML report
  annotate_confidence_bar: true  # Draw confidence bars under coins
  banner_height_px: 60           # Height of summary banner
EOF

echo -e "${GREEN}✓ config.yaml created${NC}"

# ============================================================================
# CREATE PYTHON MODULE FILES
# ============================================================================
echo -e "${GREEN}[4/6] Creating Python modules...${NC}"

# __init__.py files
touch "$PROJECT_ROOT/src/__init__.py"
touch "$PROJECT_ROOT/src/utils/__init__.py"
touch "$PROJECT_ROOT/tests/__init__.py"

# utils/image_io.py
cat > "$PROJECT_ROOT/src/utils/image_io.py" << 'EOF'
"""Image Input/Output utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def load_image(path: str) -> np.ndarray:
    """
    Load image from disk with validation.
    
    Args:
        path: Path to image file
        
    Returns:
        Image as BGR numpy array
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format unsupported or image is empty
    """
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
    """Save image to disk, creating parent directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)


def validate_image_size(img: np.ndarray, min_size: Tuple[int, int] = (100, 100)) -> bool:
    """Validate image meets minimum size requirements."""
    h, w = img.shape[:2]
    return h >= min_size[0] and w >= min_size[1]
EOF

# utils/geometry.py
cat > "$PROJECT_ROOT/src/utils/geometry.py" << 'EOF'
"""Geometry utilities for circle operations."""

import numpy as np
from typing import List, Tuple

Circle = Tuple[int, int, int]


def circle_area(r: int) -> float:
    """Compute area of circle."""
    return np.pi * r * r


def circle_iou(c1: Circle, c2: Circle) -> float:
    """
    Compute Intersection over Union for two circles.
    
    Args:
        c1, c2: (x, y, r) tuples
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        smaller_area = np.pi * min(r1, r2)**2
        larger_area = np.pi * max(r1, r2)**2
        return smaller_area / larger_area
    
    # Partial overlap - compute lens area
    a1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    a2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    intersection = a1 + a2 - a3
    union = np.pi * (r1**2 + r2**2) - intersection
    return intersection / union if union > 0 else 0.0


def nms_circles(circles: List[Circle], iou_threshold: float = 0.3) -> List[Circle]:
    """Apply Non-Maximum Suppression to remove overlapping circles."""
    if len(circles) == 0:
        return []
    
    # Sort by radius descending (larger circles first)
    sorted_circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept = []
    
    for candidate in sorted_circles:
        suppress = False
        for kept_circle in kept:
            if circle_iou(candidate, kept_circle) > iou_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(candidate)
    
    return kept
EOF

# utils/color_utils.py
cat > "$PROJECT_ROOT/src/utils/color_utils.py" << 'EOF'
"""Color space utilities for feature extraction."""

import cv2
import numpy as np
from typing import Tuple, Dict


def bgr_to_hsv_stats(roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute HSV statistics over masked region."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_bool = mask > 0
    
    h = hsv[:, :, 0][mask_bool].astype(float)
    s = hsv[:, :, 1][mask_bool].astype(float)
    v = hsv[:, :, 2][mask_bool].astype(float)
    
    return {
        'hue_mean': float(np.mean(h)) if len(h) > 0 else 0,
        'hue_std': float(np.std(h)) if len(h) > 0 else 0,
        'sat_mean': float(np.mean(s)) if len(s) > 0 else 0,
        'sat_std': float(np.std(s)) if len(s) > 0 else 0,
        'val_mean': float(np.mean(v)) if len(v) > 0 else 0,
        'val_std': float(np.std(v)) if len(v) > 0 else 0,
    }


def bgr_to_lab_stats(roi: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute LAB statistics over masked region."""
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mask_bool = mask > 0
    
    L = lab[:, :, 0][mask_bool].astype(float)
    a = lab[:, :, 1][mask_bool].astype(float)
    b = lab[:, :, 2][mask_bool].astype(float)
    
    return {
        'lab_L_mean': float(np.mean(L)) if len(L) > 0 else 0,
        'lab_a_mean': float(np.mean(a)) if len(a) > 0 else 0,
        'lab_b_mean': float(np.mean(b)) if len(b) > 0 else 0,
        'lab_L_std': float(np.std(L)) if len(L) > 0 else 0,
        'lab_a_std': float(np.std(a)) if len(a) > 0 else 0,
        'lab_b_std': float(np.std(b)) if len(b) > 0 else 0,
    }


def compute_hue_histogram(roi: np.ndarray, mask: np.ndarray, bins: int = 18) -> Dict:
    """Compute normalized hue histogram over masked region."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180]).flatten()
    
    total = hist.sum()
    if total > 0:
        hist = hist / total
    
    dom_bin = int(np.argmax(hist))
    dom_hue = (dom_bin * 180 / bins) + (180 / bins / 2)
    
    # Compute entropy
    hist_nonzero = hist[hist > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero))) if len(hist_nonzero) > 0 else 0
    
    return {
        'hue_hist': hist.tolist(),
        'dominant_hue': float(dom_hue),
        'hist_entropy': entropy,
    }
EOF

# Create placeholder phase files (actual implementation from phase_details.md)
for phase in phase1_preprocessing phase2_detection phase3_segmentation phase4_features phase5_classification phase6_output; do
    cat > "$PROJECT_ROOT/src/${phase}.py" << EOF
\"\"\"Phase ${phase#phase} module - Coin Counting System.\"\"\"
# Full implementation should be added here based on phase_details.md
# This is a placeholder - replace with actual code

class ${phase#phase^}:
    def __init__(self, config: dict, debug: bool = False):
        self.config = config
        self.debug = debug
    
    def run(self, *args, **kwargs):
        \"\"\"Execute phase processing.\"\"\"
        pass
EOF
done

echo -e "${GREEN}✓ Python modules created${NC}"

# ============================================================================
# CREATE SAMPLE TEST IMAGES (Synthetic)
# ============================================================================
echo -e "${GREEN}[5/6] Creating sample test images...${NC}"

cat > "$PROJECT_ROOT/data/generate_sample_images.py" << 'EOF'
"""Generate synthetic sample images for testing."""

import cv2
import numpy as np
from pathlib import Path

def create_single_coin_image(output_path: str, color: tuple, radius: int, 
                              position: tuple, background: tuple = (240, 240, 240)):
    """Create a single coin image."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = background
    cv2.circle(img, position, radius, color, -1)
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

def create_mixed_coins_image(output_path: str):
    """Create image with multiple coins."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    
    # 1c (copper, small)
    cv2.circle(img, (150, 150), 45, (50, 100, 200), -1)
    # 10c (gold, medium)
    cv2.circle(img, (350, 150), 50, (50, 165, 210), -1)
    # 50c (gold, large)
    cv2.circle(img, (550, 150), 60, (50, 165, 210), -1)
    # €1 (bicolor, largest)
    cv2.circle(img, (400, 400), 65, (180, 140, 220), -1)
    
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

def create_touching_coins(output_path: str):
    """Create image with touching coins."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    
    cv2.circle(img, (200, 250), 60, (50, 100, 200), -1)
    cv2.circle(img, (270, 250), 60, (50, 165, 210), -1)
    
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

def create_low_light_image(output_path: str):
    """Create low-light simulation."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (80, 80, 80)
    cv2.circle(img, (250, 250), 60, (50, 100, 200), -1)
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

if __name__ == "__main__":
    sample_dir = Path(__file__).parent / "sample_images"
    sample_dir.mkdir(exist_ok=True)
    
    print("Generating sample images...")
    create_single_coin_image(str(sample_dir / "single_coin.jpg"), (50, 100, 200), 60, (250, 250))
    create_mixed_coins_image(str(sample_dir / "mixed_coins.jpg"))
    create_touching_coins(str(sample_dir / "touching_coins.jpg"))
    create_low_light_image(str(sample_dir / "low_light.jpg"))
    print("✓ Sample images created")
EOF

python "$PROJECT_ROOT/data/generate_sample_images.py"

# ============================================================================
# CREATE MAIN.PY
# ============================================================================
echo -e "${GREEN}[6/6] Creating main.py...${NC}"

cat > "$PROJECT_ROOT/main.py" << 'EOF'
#!/usr/bin/env python3
"""
Coin Counting System - Main Entry Point
Usage: python main.py --image path/to/coins.jpg [--debug] [--config config.yaml]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Import phase modules (will be implemented)
from src.phase1_preprocessing import Preprocessor
from src.phase2_detection import CoinDetector
from src.phase3_segmentation import CoinSegmenter
from src.phase4_features import FeatureExtractor
from src.phase5_classification import CoinClassifier
from src.phase6_output import ResultExporter


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Coin Counting System')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--debug', '-d', action='store_true', help='Save intermediate outputs')
    parser.add_argument('--output-dir', '-o', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['debug'] = args.debug
    config['output_dir'] = args.output_dir
    
    print(f"\n🔍 Processing: {args.image}")
    print(f"⚙️  Config: {args.config}")
    start_time = time.time()
    
    try:
        # Phase 1: Preprocessing
        print("\n  📷 Phase 1: Preprocessing...")
        preprocessor = Preprocessor(config.get('preprocessing', {}), debug=args.debug)
        phase1_result = preprocessor.run(args.image)
        
        # Phase 2: Detection
        print("  🔘 Phase 2: Detecting circles with CHT...")
        detector = CoinDetector(config.get('detection', {}), debug=args.debug)
        phase2_result = detector.run(phase1_result)
        
        if phase2_result.get('count', 0) == 0:
            print("  ❌ No coins detected!")
            return
        
        print(f"     Found {phase2_result['count']} coins")
        
        # Phase 3: Segmentation
        print("  ✂️  Phase 3: Segmenting coins...")
        segmenter = CoinSegmenter(config.get('segmentation', {}), debug=args.debug)
        phase3_result = segmenter.run(phase1_result, phase2_result)
        
        # Phase 4: Features
        print("  📊 Phase 4: Extracting features...")
        extractor = FeatureExtractor(config.get('features', {}), debug=args.debug)
        phase4_result = extractor.run(phase1_result, phase2_result, phase3_result)
        
        # Phase 5: Classification
        print("  🏷️  Phase 5: Classifying coins...")
        classifier = CoinClassifier(config.get('classification', {}))
        phase5_result = classifier.run(phase4_result)
        
        # Phase 6: Output
        print("  📄 Phase 6: Generating outputs...")
        exporter = ResultExporter(config.get('output', {}), output_dir=args.output_dir)
        all_results = {
            'phase1': phase1_result,
            'phase2': phase2_result,
            'phase3': phase3_result,
            'phase4': phase4_result,
            'phase5': phase5_result,
        }
        phase6_result = exporter.run(all_results, args.image)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Complete! Time: {elapsed:.2f}s")
        print(f"   Total value: €{phase5_result['summary']['total_eur']:.2f}")
        print(f"   Outputs saved to: {args.output_dir}/\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
EOF

chmod +x "$PROJECT_ROOT/main.py"

# ============================================================================
# CREATE README.MD
# ============================================================================
cat > "$PROJECT_ROOT/README.md" << 'EOF'
# 🪙 Coin Counting System

A classical computer vision system that detects, classifies, and counts coins using Circular Hough Transform and hand-crafted color features. **No machine learning or deep learning** — pure image processing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on sample image
python main.py --image data/sample_images/mixed_coins.jpg

# Run with debug output (saves intermediate images)
python main.py --image data/sample_images/mixed_coins.jpg --debug
