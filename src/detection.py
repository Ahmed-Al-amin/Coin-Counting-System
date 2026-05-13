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

    def detect_circles(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Run OpenCV HoughCircles on an image (preferably blurred grayscale).
        """
        # Ensure we use the 'detection' sub-config if provided, else use config directly
        cfg = self.config if 'dp' in self.config else self.config.get('detection', {})
        
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,
            dp        = cfg.get('dp', 1.2),
            minDist   = cfg.get('min_dist', 120),
            param1    = cfg.get('param1', 120),
            param2    = cfg.get('param2', 60), 
            minRadius = cfg.get('min_radius', 15),
            maxRadius = cfg.get('max_radius', 200),
        )
        return circles  # Shape: (1, N, 3) or None

    def score_circle(self, c: Circle, edges: np.ndarray) -> float:
        """
        Calculate edge strength along the circle boundary.
        Higher score means the circle better aligns with real image edges.
        """
        x, y, r = c
        mask = np.zeros(edges.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, 2)  # Thickness 2 to capture nearby edges

        # Sum of edge pixels along the perimeter
        edge_strength = np.sum(edges[mask > 0])
        return float(edge_strength)

    def validate_circle_position(self, c: Circle, edges: np.ndarray) -> bool:
        x, y, r = c
        h, w = edges.shape[:2]
        
        # Create a thin ring mask at the circle's boundary
        mask_ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask_ring, (x, y), r, 255, 3) # 3px thick ring
        
        # Count how many edge pixels fall ON the circle's boundary
        rim_edges = cv2.bitwise_and(edges, mask_ring)
        edge_count = np.count_nonzero(rim_edges)
        
        # Mathematical perimeter is 2 * pi * r
        # We require at least 35% of the perimeter to have strong edges
        expected_perimeter = 2 * np.pi * r
        return edge_count > (expected_perimeter * 0.35)

    def enforce_one_circle_per_coin(self, circles: List[Circle]) -> List[Circle]:
        """
        Final safety filter to ensure exactly one circle per physical coin.
        If centers are very close relative to radius, keeps only the larger (outer) one.
        """
        final: List[Circle] = []
        for c in circles:
            keep = True
            for f in final:
                dist = np.sqrt((c[0]-f[0])**2 + (c[1]-f[1])**2)
                
                # If centers are very close (relative to coin size) -> same physical coin
                if dist < 0.6 * max(c[2], f[2]):
                    # Keep larger radius (outer boundary)
                    if c[2] > f[2]:
                        final.remove(f)
                    else:
                        keep = False
                    break
            
            if keep:
                final.append(c)
        
        return final

    def estimate_radius_range(self, radii: List[int]) -> Optional[Tuple[int, int]]:
        """
        Estimate a robust radius range based on dominant contour sizes.
        Rejects outliers using percentiles to find the 'typical' coin size in the scene.
        """
        if len(radii) == 0:
            return None

        radii_arr = np.array(radii)

        # Remove extreme outliers using percentiles (20th to 80th)
        low = np.percentile(radii_arr, 20)
        high = np.percentile(radii_arr, 80)

        filtered = radii_arr[(radii_arr >= low) & (radii_arr <= high)]

        if len(filtered) == 0:
            return None

        median_r = np.median(filtered)

        # Allow some flexibility around the dominant size
        min_r = int(0.6 * median_r)
        max_r = int(1.5 * median_r)

        return max(15, min_r), min(300, max_r)

    def morphological_closing(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply morphological closing (dilation then erosion) to bridge broken edge segments.
        Ensures contours are extracted from continuous shapes rather than fragments.
        """
        kernel_size = self.config.get('morph_kernel', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    def detect_via_contours(self, edges: np.ndarray) -> List[Circle]:
        """
        Detect coins using contour analysis. 
        Filters by area, circularity, and radius consistency.
        """
        cfg = self.config if 'min_radius' in self.config else self.config.get('detection', {})
        min_r = cfg.get('min_radius', 15)
        max_r = cfg.get('max_radius', 200)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circles_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # 1. Area Filter: remove tiny noise
            if perimeter == 0 or area < (np.pi * min_r**2 * 0.4):
                continue
                
            # 2. Circularity Filter: 4 * pi * Area / Perimeter^2
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity > 0.55: # Allows heptagonal/octagonal coins
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                
                # 3. Radius Consistency: Enclosing circle should match contour area density
                enclosing_area = np.pi * radius**2
                if area / enclosing_area < 0.6:  # Fragmented shape
                    continue
                    
                if min_r <= radius <= max_r:
                    circles_list.append((int(x), int(y), int(radius)))
                    
        return circles_list

    @staticmethod
    def suggest_radius_range(image_shape: tuple, 
                              min_coin_fraction: float = 0.04,
                              max_coin_fraction: float = 0.15) -> Tuple[int, int]:
        """
        Auto-estimate radius range from image size with tighter constraints.
        """
        diag = np.sqrt(image_shape[0]**2 + image_shape[1]**2)
        min_r = int(diag * min_coin_fraction)
        max_r = int(diag * max_coin_fraction)
        return max(15, min_r), min(300, max_r)

    def cluster_and_select_best(self, circles: List[Circle], edges: np.ndarray) -> List[Circle]:
        """
        Group circles that belong to the same coin using centroid-based logic and select best edge alignment.
        """
        clusters: List[List[Circle]] = []

        for c in circles:
            added = False
            for cluster in clusters:
                # Fix 2: Centroid-based clustering for stability
                cx = int(np.mean([cl[0] for cl in cluster]))
                cy = int(np.mean([cl[1] for cl in cluster]))
                cr = int(np.mean([cl[2] for cl in cluster]))
                
                d = np.sqrt((c[0]-cx)**2 + (c[1]-cy)**2)
                dist_thresh = int(0.8 * cr)

                if d < dist_thresh and abs(c[2] - cr) < 0.5 * cr:
                    cluster.append(c)
                    added = True
                    break

            if not added:
                clusters.append([c])

        # Select best circle from each cluster
        final = []
        for cluster in clusters:
            # Fix 4: Keep only biggest candidates (top 2) for scoring
            cluster = sorted(cluster, key=lambda x: x[2])[-2:]
            
            best = max(cluster, key=lambda c: self.score_circle(c, edges))
            final.append(best)

        return final

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
        edges    = phase1_result['edges']
        original = phase1_result['scaled']
        
        # Access config correctly
        cfg = self.config if 'dp' in self.config else self.config.get('detection', {})
        
        # Auto-set radius range if not in config
        if 'min_radius' not in cfg:
            min_r, max_r = self.suggest_radius_range(edges.shape)
            cfg['min_radius'] = min_r
            cfg['max_radius'] = max_r
        
        # 1. Close the edges to bridge gaps before contour detection
        closed_edges = self.morphological_closing(edges)
        
        # HYBRID DETECTION (GUIDED BY CONTOURS)
        circles_list: List[Circle] = []
        
        # 2. Helper: Contour detection first to estimate scene scale
        contours = self.detect_via_contours(closed_edges)
        circles_list.extend(contours)
        
        # 3. Dynamic radius range estimation based on initial contours
        radius_range = self.estimate_radius_range([r for (_, _, r) in contours])
        if radius_range is not None:
            min_r, max_r = radius_range
            cfg['min_radius'] = min_r
            cfg['max_radius'] = max_r
        elif 'min_radius' not in cfg:
            # Fallback to static estimation if no contours found
            min_r, max_r = self.suggest_radius_range(edges.shape)
            cfg['min_radius'] = min_r
            cfg['max_radius'] = max_r
        
        # 4. Primary: CHT on bilateral image (now guided by scene scale)
        raw_circles = self.detect_circles(blurred)
        if raw_circles is not None:
            circles_list.extend([(int(x), int(y), int(r)) for x, y, r in raw_circles[0]])
        
        # Post-processing
        if len(circles_list) > 0:
            # 1. NMS for overlap (Fix 3: Increased threshold)
            nms_thresh = cfg.get('nms_iou_threshold', 0.4)
            circles = nms_circles(circles_list, nms_thresh)
            
            # 2. Cluster circles by center and select best (centroid-based)
            circles = self.cluster_and_select_best(circles, edges)
            
            # 3. Enforce one circle per coin (Outermost boundary filter)
            circles = self.enforce_one_circle_per_coin(circles)
            
            # 4. Position validation (enough edges)
            circles = [c for c in circles if self.validate_circle_position(c, edges)]
            
            # 5. Boundary filtering
            margin_ratio = cfg.get('boundary_margin', 0.5)
            circles = self.filter_boundary_circles(circles, edges.shape, margin_ratio)
        else:
            circles = []
        
        viz = self.visualize_detections(original, circles)
        
        if self.debug:
            save_image(viz, f"{self.output_dir}/detections.png")
        
        return {
            "circles":       circles,           # List[(x, y, r)]
            "count":         len(circles),
            "visualization": viz,
            "scale_factor":  phase1_result['scale_factor'],
            "detection_mode": "HYBRID"
        }
