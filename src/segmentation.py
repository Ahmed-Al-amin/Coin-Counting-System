"""Phase 3: Segmentation module - Coin Counting System."""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict
from src.utils.image_io import save_image

Circle = Tuple[int, int, int]

class CoinSegmenter:
    def __init__(self, config: dict, debug: bool = False,
                 output_dir: str = "outputs/intermediate/phase3"):
        """
        Initialize the CoinSegmenter with configuration.
        
        Args:
            config: Segmentation configuration dictionary
            debug: Whether to save intermediate debug images
            output_dir: Directory to save debug images
        """
        self.config = config
        self.debug = debug
        self.output_dir = output_dir
        self.target_size = config.get('target_size', 128)
        self.padding_ratio = config.get('padding_ratio', 0.05)
        
        if self.debug:
            os.makedirs(self.output_dir, exist_ok=True)

    def crop_roi(self, image: np.ndarray, x: int, y: int, r: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract a square bounding box around the coin with optional padding.
        
        Padding prevents masking artifacts at exact circle boundary by
        including a small margin (5% of radius by default).
        
        Clamps to image boundaries to handle partially out-of-frame coins.
        """
        pad = int(r * self.padding_ratio)
        r_padded = r + pad
        
        # Compute bounding box with clamping
        x1 = max(0, x - r_padded)
        y1 = max(0, y - r_padded)
        x2 = min(image.shape[1], x + r_padded)
        y2 = min(image.shape[0], y + r_padded)
        
        return image[y1:y2, x1:x2].copy(), (x1, y1)  # ROI + top-left offset

    def apply_circular_mask(self, roi: np.ndarray, 
                             cx_local: int, cy_local: int, 
                             r: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply a circular binary mask to isolate the coin from background.
        
        Returns:
            masked_roi: BGR image with background set to black (0,0,0)
            mask:       Binary mask (255=coin, 0=background)
        """
        h, w = roi.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw filled white circle at local coordinates
        cv2.circle(mask, (cx_local, cy_local), r, 255, -1)
        
        # Optional: apply slight erosion to avoid boundary pixels
        erode_px = self.config.get('mask_erosion_px', 2)
        if erode_px > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                               (erode_px*2+1, erode_px*2+1))
            mask = cv2.erode(mask, kernel)
        
        # Apply mask: background becomes pure black
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)
        
        return masked_roi, mask

    def normalize_roi(self, masked_roi: np.ndarray, 
                       mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize coin ROI to target_size x target_size.
        
        Using INTER_AREA for downscaling (antialiasing), INTER_CUBIC for 
        upscaling (smooth interpolation).
        """
        h, w = masked_roi.shape[:2]
        t = self.target_size
        
        interp = cv2.INTER_AREA if max(h, w) > t else cv2.INTER_CUBIC
        
        norm_roi = cv2.resize(masked_roi, (t, t), interpolation=interp)
        norm_mask = cv2.resize(mask, (t, t), interpolation=cv2.INTER_NEAREST)
        
        return norm_roi, norm_mask

    def segment_coin(self, image: np.ndarray, 
                     circle: Circle, 
                     coin_index: int) -> Dict:
        """
        Full segmentation pipeline for a single coin.
        Returns a metadata dict containing all representations.
        """
        x, y, r = circle
        
        # Step 1: Extract ROI
        roi, (x1, y1) = self.crop_roi(image, x, y, r)
        
        # Step 2: Compute local center in cropped ROI coordinate system
        cx_local = x - x1
        cy_local = y - y1
        
        # Step 3: Apply circular mask
        masked_roi, mask = self.apply_circular_mask(roi, cx_local, cy_local, r)
        
        # Step 4: Normalize
        norm_roi, norm_mask = self.normalize_roi(masked_roi, mask)
        
        return {
            "index":       coin_index,
            "circle":      circle,           # (x, y, r) in full image coords
            "roi":         roi,              # Raw cropped square
            "masked_roi":  masked_roi,       # Circle-masked coin
            "mask":        mask,             # Binary mask
            "norm_roi":    norm_roi,         # 128x128 normalized coin
            "norm_mask":   norm_mask,        # 128x128 normalized mask
            "local_center": (cx_local, cy_local),
        }

    def run(self, phase1_result: dict, phase2_result: dict) -> dict:
        """
        Segment all detected coins.
        Input:  Phase 1 result (for original color image) + Phase 2 result (circles)
        Output: List of coin segment dicts
        """
        image = phase1_result['scaled']   # Full color image
        circles = phase2_result['circles']  # List[(x, y, r)]
        
        segments = []
        for i, circle in enumerate(circles):
            seg = self.segment_coin(image, circle, i)
            segments.append(seg)
            
            if self.debug:
                os.makedirs(f"{self.output_dir}/coins", exist_ok=True)
                save_image(seg['norm_roi'],  
                           f"{self.output_dir}/coins/coin_{i:03d}_norm.png")
                save_image(seg['masked_roi'], 
                           f"{self.output_dir}/coins/coin_{i:03d}_masked.png")
        
        if self.debug:
            grid = self._make_coin_grid(segments)
            save_image(grid, f"{self.output_dir}/all_coins_grid.png")
        
        return {
            "segments":   segments,
            "count":      len(segments),
        }

    def _make_coin_grid(self, segments: list, cols: int = 8) -> np.ndarray:
        """Arrange all normalized coins into a grid image for overview."""
        if not segments:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            
        t = self.target_size
        n = len(segments)
        rows = (n + cols - 1) // cols
        
        grid = np.zeros((rows * t, cols * t, 3), dtype=np.uint8)
        for i, seg in enumerate(segments):
            r, c = divmod(i, cols)
            grid[r*t:(r+1)*t, c*t:(c+1)*t] = seg['norm_roi']
        
        return grid
