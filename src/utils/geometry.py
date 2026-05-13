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
