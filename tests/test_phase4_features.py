import pytest
import cv2
import numpy as np
from src.features import FeatureExtractor


CONFIG = {
    'size_thresholds': {
        'tiny': 0.025,
        'small': 0.040,
        'medium': 0.055,
        'large': 0.070
    },
    'bicolor_hue_std_thresh': 22,
}


@pytest.fixture
def extractor():
    return FeatureExtractor(config=CONFIG, debug=False)


def make_coin_roi(color_bgr, size=128, radius=60):
    """Create synthetic coin ROI + mask"""
    roi = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)

    cv2.circle(roi, (size // 2, size // 2), radius, color_bgr, -1)
    cv2.circle(mask, (size // 2, size // 2), radius, 255, -1)

    return roi, mask


# ---------------- SIZE ----------------

def test_size_feature_keys(extractor):
    feat = extractor.compute_size_feature(50, 1000.0)
    assert {'radius_px', 'rel_size', 'rel_diam', 'size_bin'}.issubset(feat.keys())


def test_size_bin_tiny(extractor):
    feat = extractor.compute_size_feature(20, 1000.0)  # 0.02
    assert feat['size_bin'] == 'tiny'


def test_size_bin_xlarge(extractor):
    feat = extractor.compute_size_feature(80, 1000.0)  # 0.08
    assert feat['size_bin'] == 'xlarge'


# ---------------- HSV ----------------

def test_hsv_features_masked(extractor):
    roi, mask = make_coin_roi((30, 165, 210))  # gold-ish
    feats = extractor.compute_hsv_features(roi, mask)

    required = {
        'hue_mean', 'hue_std',
        'sat_mean', 'sat_std',
        'val_mean', 'val_std'
    }

    assert required.issubset(feats.keys())
    assert 0 <= feats['hue_mean'] <= 180
    assert 0 <= feats['sat_mean'] <= 255


# ---------------- LAB ----------------

def test_lab_features(extractor):
    roi, mask = make_coin_roi((30, 100, 180))
    feats = extractor.compute_lab_features(roi, mask)

    required = {'lab_L_mean', 'lab_a_mean', 'lab_b_mean'}
    assert required.issubset(feats.keys())
    assert 0 <= feats['lab_L_mean'] <= 255


# ---------------- HISTOGRAM ----------------

def test_hue_histogram_normalized(extractor):
    roi, mask = make_coin_roi((20, 130, 200))
    hist_feats = extractor.compute_hue_histogram(roi, mask)

    hist = hist_feats['hue_hist']
    assert abs(sum(hist) - 1.0) < 0.01


def test_histogram_entropy_uniform(extractor):
    roi, mask = make_coin_roi((0, 128, 255))  # orange
    hist_feats = extractor.compute_hue_histogram(roi, mask, bins=18)

    assert hist_feats['hist_entropy'] < 3.0


# ---------------- COLOR ----------------

def test_copper_color_group(extractor):
    roi, mask = make_coin_roi((50, 100, 180))
    hsv_f = extractor.compute_hsv_features(roi, mask)
    lab_f = extractor.compute_lab_features(roi, mask)

    group = extractor.classify_color_group(hsv_f, lab_f)

    assert group in {'COPPER', 'GOLD', 'BICOLOR', 'SILVER', 'UNKNOWN'}


def test_silver_color_group(extractor):
    roi, mask = make_coin_roi((180, 180, 180))  # gray
    hsv_f = extractor.compute_hsv_features(roi, mask)
    lab_f = extractor.compute_lab_features(roi, mask)

    group = extractor.classify_color_group(hsv_f, lab_f)

    assert group == 'SILVER'


# ---------------- FULL VECTOR ----------------

def test_feature_vector_complete(extractor):
    roi, mask = make_coin_roi((30, 100, 180))

    segment = {
        'index': 0,
        'circle': (200, 200, 50),
        'norm_roi': roi,
        'norm_mask': mask,
    }

    feat = extractor.build_feature_vector(segment, image_diagonal=1000.0)

    expected = {
        'coin_index', 'circle',
        'rel_size', 'size_bin',
        'hue_mean', 'sat_mean',
        'lab_a_mean', 'lab_b_mean',
        'color_group',
        'hue_hist', 'dominant_hue'
    }

    assert expected.issubset(feat.keys())