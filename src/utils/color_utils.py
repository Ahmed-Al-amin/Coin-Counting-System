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
