"""Phase 2: Detection module - Coin Counting System."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.utils.image_io import save_image
from src.utils.geometry import nms_circles

Circle = Tuple[int, int, int]  # (x, y, r)

class CoinDetector:
    def __init__(self, config: dict, debug: bool = False,
                 output_dir: str = "outputs/intermediate/phase2"):
        """
        Initialize the CoinDetector with configuration.
        
        Args:
            config: Detection configuration dictionary
            debug: Whether to save intermediate debug images
            output_dir: Directory to save debug images
        """
        self.config     = config
        self.debug      = debug
        self.output_dir = output_dir

    def detect_circles(self, blurred_gray: np.ndarray) -> np.ndarray:
        """
        Run OpenCV HoughCircles on a blurred grayscale image.
        """
        # Ensure we use the 'detection' sub-config if provided, else use config directly
        cfg = self.config if 'dp' in self.config else self.config.get('detection', {})
        
        circles = cv2.HoughCircles(
            blurred_gray,
            cv2.HOUGH_GRADIENT,
            dp        = cfg.get('dp', 1.2),
            minDist   = cfg.get('min_dist', 30),
            param1    = cfg.get('param1', 80),
            param2    = cfg.get('param2', 30),
            minRadius = cfg.get('min_radius', 15),
            maxRadius = cfg.get('max_radius', 200),
        )
        return circles  # Shape: (1, N, 3) or None

    @staticmethod
    def suggest_radius_range(image_shape: tuple, 
                              min_coin_fraction: float = 0.03,
                              max_coin_fraction: float = 0.25) -> Tuple[int, int]:
        """
        Auto-estimate radius range from image size.
        """
        diag = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        min_r = int(diag * min_coin_fraction)
        max_r = int(diag * max_coin_fraction)
        return max(10, min_r), min(500, max_r)

    def filter_boundary_circles(self, circles: List[Circle], 
                                  image_shape: tuple,
                                  margin_ratio: float = 0.5) -> List[Circle]:
        """
        Remove circles whose center is too close to the image boundary.
        """
        h, w = image_shape[:2]
        filtered = []
        for (x, y, r) in circles:
            margin = int(r * margin_ratio)
            if (x - margin >= 0 and x + margin < w and
                y - margin >= 0 and y + margin < h):
                filtered.append((x, y, r))
        return filtered

    def visualize_detections(self, original: np.ndarray, 
                              circles: List[Circle]) -> np.ndarray:
        """Draw detected circles on a copy of the original image."""
        viz = original.copy()
        for i, (x, y, r) in enumerate(circles):
            # Outer circle
            cv2.circle(viz, (x, y), r, (0, 255, 0), 2)
            # Center dot
            cv2.circle(viz, (x, y), 3, (0, 0, 255), -1)
            # Index label
            cv2.putText(viz, str(i), (x - 8, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return viz

    def run(self, phase1_result: dict) -> dict:
        """
        Execute full detection pipeline.
        Input:  phase1_result dict from Preprocessor.run()
        Output: dict with 'circles' (List[Circle]) and 'visualization'
        """
        blurred  = phase1_result['blurred']
        original = phase1_result['scaled']
        
        # Access config correctly
        cfg = self.config if 'dp' in self.config else self.config.get('detection', {})
        
        # Auto-set radius range if not in config
        if 'min_radius' not in cfg:
            min_r, max_r = self.suggest_radius_range(blurred.shape)
            cfg['min_radius'] = min_r
            cfg['max_radius'] = max_r
        
        raw_circles = self.detect_circles(blurred)
        
        if raw_circles is None:
            circles = []
        else:
            circles_list = [(int(x), int(y), int(r)) 
                            for x, y, r in raw_circles[0]]
            
            # Use NMS from geometry utils
            nms_thresh = cfg.get('nms_iou_threshold', 0.3)
            circles = nms_circles(circles_list, nms_thresh)
            
            # Boundary filtering
            margin_ratio = cfg.get('boundary_margin', 0.5)
            circles = self.filter_boundary_circles(circles, blurred.shape, margin_ratio)
        
        viz = self.visualize_detections(original, circles)
        
        if self.debug:
            save_image(viz, f"{self.output_dir}/detections.png")
        
        return {
            "circles":       circles,           # List[(x, y, r)]
            "count":         len(circles),
            "visualization": viz,
            "scale_factor":  phase1_result['scale_factor'],
        }
