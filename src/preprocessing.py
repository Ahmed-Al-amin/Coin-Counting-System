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

    def bilateral_filter(self, gray: np.ndarray) -> np.ndarray:
        """
        Smooth internal textures while preserving sharp boundaries.
        Bilateral filtering is nonlinear and edge-preserving, making it ideal
        for coins where internal patterns should be flattened.
        """
        d = self.config.get('bilateral_d', 9)
        sigma_color = self.config.get('bilateral_sigma_color', 75)
        sigma_space = self.config.get('bilateral_sigma_space', 75)
        return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

    def sobel_magnitude(self, blurred: np.ndarray) -> np.ndarray:
        """
        Detect edges using Sobel magnitude. 
        Includes normalization and softer thresholding for hybrid detection.
        """
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        
        # Normalize to 0-255 range
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = np.uint8(edges)
        
        # Apply softer binary threshold
        _, edges_binary = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
        return edges_binary

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
        blurred = self.bilateral_filter(clahe)
        edges = self.sobel_magnitude(blurred)
        
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
