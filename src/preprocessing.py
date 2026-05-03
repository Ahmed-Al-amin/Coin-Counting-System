"""Phase 1: Preprocessing module - Coin Counting System."""

import cv2
import numpy as np
from src.utils.image_io import load_image, save_image

class Preprocessor:
    def __init__(self, config: dict, debug: bool = False, output_dir: str = "outputs/intermediate/phase1"):
        """
        Initialize the Preprocessor with configuration.
        
        Args:
            config: Preprocessing configuration dictionary
            debug: Whether to save intermediate debug images
            output_dir: Directory to save debug images
        """
        self.config = config
        self.debug = debug
        self.output_dir = output_dir

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR to grayscale. Coins have circular shape best captured in intensity."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_clahe(self, gray: np.ndarray) -> np.ndarray:
        """
        Enhance local contrast without blowing out highlights.
        CLAHE divides the image into tiles and equalizes each tile's
        histogram independently, preventing over-amplification of noise.
        """
        clip_limit = self.config.get('clahe_clip_limit', 2.0)
        tile_grid_size = tuple(self.config.get('clahe_tile_size', [8, 8]))
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(gray)

    def gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """
        Smooth high-frequency noise before edge detection.
        Kernel size must be odd. Larger kernels = more smoothing.
        Typical: 5×5 for clean images, 9×9 for noisy ones.
        """
        kernel_size = tuple(self.config.get('blur_kernel', [5, 5]))
        sigma = self.config.get('blur_sigma', 1.5)
        return cv2.GaussianBlur(gray, kernel_size, sigma)

    def canny_edges(self, blurred: np.ndarray) -> np.ndarray:
        """
        Detect edges using the Canny algorithm.
        Auto-thresholding using Otsu's method for adaptability.
        """
        # Auto-compute thresholds from image statistics
        otsu_thresh, _ = cv2.threshold(blurred, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_ratio = self.config.get('canny_low_ratio', 0.5)
        high_ratio = self.config.get('canny_high_ratio', 1.0)
        
        low = otsu_thresh * low_ratio
        high = otsu_thresh * high_ratio
        
        return cv2.Canny(blurred, int(low), int(high))

    def normalize_scale(self, image: np.ndarray) -> np.ndarray:
        """
        Resize very large images to a standard working resolution.
        This keeps CHT radius ranges consistent across cameras.
        Target: longest side = 1024px (preserves aspect ratio).
        """
        max_dim = self.config.get('max_working_dim', 1024)
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        return cv2.resize(image, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def run(self, image_path: str) -> dict:
        """Execute full preprocessing pipeline, return all intermediate results."""
        original = load_image(image_path)
        scaled = self.normalize_scale(original)
        gray = self.to_grayscale(scaled)
        clahe = self.apply_clahe(gray)
        blurred = self.gaussian_blur(clahe)
        edges = self.canny_edges(blurred)
        
        if self.debug:
            save_image(gray, f"{self.output_dir}/gray.png")
            save_image(clahe, f"{self.output_dir}/clahe.png")
            save_image(blurred, f"{self.output_dir}/blurred.png")
            save_image(edges, f"{self.output_dir}/edges.png")
        
        return {
            "original": original,
            "scaled": scaled,
            "gray": gray,
            "clahe": clahe,
            "blurred": blurred,
            "edges": edges,
            "scale_factor": scaled.shape[1] / original.shape[1]
        }
