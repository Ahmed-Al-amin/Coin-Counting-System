import numpy as np

def circle_iou(c1, c2):
    """Calculate Intersection over Union of two circles."""
    x1, y1, r1 = c1
    x2, y2, r2 = c2
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    if d >= r1 + r2: return 0.0
    if d <= abs(r1 - r2):
        # smaller_r = min(r1, r2)
        return 1.0 # One is inside the other
        
    # Standard circle intersection area formula
    r1sq, r2sq = r1**2, r2**2
    dsq = d**2
    part1 = r1sq * np.arccos((dsq + r1sq - r2sq) / (2 * d * r1))
    part2 = r2sq * np.arccos((dsq + r2sq - r1sq) / (2 * d * r2))
    part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    intersection = part1 + part2 - part3
    
    union = (np.pi * r1sq) + (np.pi * r2sq) - intersection
    return intersection / union

def nms_circles(circles, threshold=0.4):
    """Non-Maximum Suppression for circles."""
    if not circles: return []
    
    # Sort by radius descending (prefer larger circles for coins)
    circles = sorted(circles, key=lambda x: x[2], reverse=True)
    keep = []
    
    while circles:
        best = circles.pop(0)
        keep.append(best)
        circles = [c for c in circles if circle_iou(best, c) < threshold]
        
    return keep
