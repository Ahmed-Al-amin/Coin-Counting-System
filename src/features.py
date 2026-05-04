import cv2
import numpy as np
from typing import Dict, List


class FeatureExtractor:
    def __init__(self, config: dict, debug: bool = False,
                 output_dir: str = "outputs/intermediate/phase4"):
        self.config = config
        self.debug = debug
        self.output_dir = output_dir

    # ---------------- SIZE ----------------
    def compute_size_feature(self, radius_px: int, image_diagonal_px: float) -> Dict:
        rel_size = radius_px / image_diagonal_px
        rel_diam = rel_size * 2

        thresholds = self.config.get('size_thresholds', {
            'tiny': 0.025,
            'small': 0.040,
            'medium': 0.055,
            'large': 0.070,
        })

        if rel_size < thresholds['tiny']:
            size_bin = 'tiny'
        elif rel_size < thresholds['small']:
            size_bin = 'small'
        elif rel_size < thresholds['medium']:
            size_bin = 'medium'
        elif rel_size < thresholds['large']:
            size_bin = 'large'
        else:
            size_bin = 'xlarge'

        return {
            'radius_px': radius_px,
            'rel_size': rel_size,
            'rel_diam': rel_diam,
            'size_bin': size_bin,
        }

    # ---------------- HSV ----------------
    def compute_hsv_features(self, norm_roi: np.ndarray, norm_mask: np.ndarray) -> Dict:
        hsv = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2HSV)
        mask = norm_mask > 0

        h = hsv[:, :, 0][mask].astype(float)
        s = hsv[:, :, 1][mask].astype(float)
        v = hsv[:, :, 2][mask].astype(float)

        return {
            'hue_mean': float(np.mean(h)),
            'hue_std': float(np.std(h)),
            'sat_mean': float(np.mean(s)),
            'sat_std': float(np.std(s)),
            'val_mean': float(np.mean(v)),
            'val_std': float(np.std(v)),
        }

    # ---------------- LAB ----------------
    def compute_lab_features(self, norm_roi: np.ndarray, norm_mask: np.ndarray) -> Dict:
        lab = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2LAB)
        mask = norm_mask > 0

        L = lab[:, :, 0][mask].astype(float)
        a = lab[:, :, 1][mask].astype(float)
        b = lab[:, :, 2][mask].astype(float)

        return {
            'lab_L_mean': float(np.mean(L)),
            'lab_a_mean': float(np.mean(a)),
            'lab_b_mean': float(np.mean(b)),
            'lab_L_std': float(np.std(L)),
            'lab_a_std': float(np.std(a)),
            'lab_b_std': float(np.std(b)),
        }

    # ---------------- HISTOGRAM ----------------
    def compute_hue_histogram(self, norm_roi: np.ndarray, norm_mask: np.ndarray, bins: int = 18) -> Dict:
        hsv = cv2.cvtColor(norm_roi, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0], norm_mask, [bins], [0, 180]).flatten()

        total = hist.sum()
        if total > 0:
            hist = hist / total

        dom_bin = int(np.argmax(hist))
        dom_hue = (dom_bin * 180 / bins) + (180 / bins / 2)

        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero)) if len(hist_nonzero) > 0 else 0.0

        return {
            'hue_hist': hist.tolist(),
            'dominant_hue': float(dom_hue),
            'hist_entropy': float(entropy),
        }

    # ---------------- COLOR GROUP ----------------
    def classify_color_group(self, hsv_feats: Dict, lab_feats: Dict) -> str:
        hue_mean = hsv_feats['hue_mean']
        hue_std = hsv_feats['hue_std']
        sat_mean = hsv_feats['sat_mean']
        lab_a = lab_feats['lab_a_mean']
        lab_b = lab_feats['lab_b_mean']

        if hue_std > self.config.get('bicolor_hue_std_thresh', 22):
            return 'BICOLOR'

        if 8 <= hue_mean <= 25 and sat_mean > 70 and lab_a > 135:
            return 'COPPER'

        if 15 <= hue_mean <= 40 and sat_mean > 30 and lab_b > 138:
            return 'GOLD'

        if sat_mean < 45:
            return 'SILVER'

        return 'UNKNOWN'

    # ---------------- FULL FEATURE ----------------
    def build_feature_vector(self, segment: Dict, image_diagonal: float) -> Dict:
        norm_roi = segment['norm_roi']
        norm_mask = segment['norm_mask']
        x, y, r = segment['circle']

        size_feats = self.compute_size_feature(r, image_diagonal)
        hsv_feats = self.compute_hsv_features(norm_roi, norm_mask)
        lab_feats = self.compute_lab_features(norm_roi, norm_mask)
        hist_feats = self.compute_hue_histogram(norm_roi, norm_mask)

        color_group = self.classify_color_group(hsv_feats, lab_feats)

        return {
            "coin_index": segment['index'],
            "circle": segment['circle'],
            **size_feats,
            **hsv_feats,
            **lab_feats,
            **hist_feats,
            "color_group": color_group,
        }

    # ---------------- RUN ----------------
    def run(self, phase1_result: dict,
            phase2_result: dict,
            phase3_result: dict) -> dict:

        h, w = phase1_result['scaled'].shape[:2]
        image_diagonal = np.sqrt(h**2 + w**2)

        features = []
        for seg in phase3_result['segments']:
            feat = self.build_feature_vector(seg, image_diagonal)
            features.append(feat)

        return {
            "features": features,
            "image_diagonal": image_diagonal
        }