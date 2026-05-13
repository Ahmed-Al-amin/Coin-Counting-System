"""Image Input/Output utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


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
