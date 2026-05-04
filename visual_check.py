import cv2
import numpy as np
import os
from src.phase3_segmentation import CoinSegmenter

# 1. Create a fake "Phase 1" image (Dark table background)
# We use a dark grey background to simulate a table
image = np.full((500, 500, 3), (50, 50, 50), dtype=np.uint8)

# Draw 4 fake "coins" on the table
# format: cv2.circle(image, (x, y), radius, BGR_color, -1)
cv2.circle(image, (100, 100), 60, (30, 165, 210), -1)  # Gold coin top-left
cv2.circle(image, (400, 100), 55, (192, 192, 192), -1) # Silver coin top-right
cv2.circle(image, (100, 400), 65, (40, 100, 150), -1)  # Bronze coin bottom-left
cv2.circle(image, (400, 400), 80, (30, 165, 210), -1)  # Big Gold coin bottom-right (partially out of bounds to test edge case!)

# 2. Create fake "Phase 2" circle detections (x, y, radius)
mock_circles = [
    (100, 100, 60),
    (400, 100, 55),
    (100, 400, 65),
    (400, 400, 80)
]

# 3. Setup Phase 1 and Phase 2 mock results
phase1_result = {'scaled': image}
phase2_result = {'circles': mock_circles}

# 4. Initialize Segmenter with DEBUG = TRUE
config = {
    'target_size': 128,
    'padding_ratio': 0.05,
    'mask_erosion_px': 2,
}

# Notice debug=True here!
segmenter = CoinSegmenter(config=config, debug=True, output_dir="outputs/intermediate/phase3")

print("Running segmentation pipeline...")
results = segmenter.run(phase1_result, phase2_result)

print(f"Success! Segmented {results['count']} coins.")
print(f"Check the 'outputs/intermediate/phase3/' folder to see your images!")