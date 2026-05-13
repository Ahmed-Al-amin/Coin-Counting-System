import pytest
import numpy as np
from src.classification import CoinClassifier


# ---------------- CONFIG ----------------
CONFIG = {
    'size_thresholds': {
        'small_medium': 0.035,
        'medium_large': 0.055,
    },
    'low_confidence_threshold': 0.50
}


# ---------------- FIXTURE ----------------
@pytest.fixture
def classifier():
    return CoinClassifier(CONFIG)


# ---------------- HELPERS ----------------
def make_features(rel_size, color_group='COPPER', coin_index=0,
                  sat_mean=100, hist_entropy=2.0, lab_L_mean=150):
    return {
        'coin_index': coin_index,
        'circle': (100, 100, 50),  # (x, y, radius)
        'color_group': color_group,
        'rel_size': rel_size,
        'sat_mean': sat_mean,
        'hist_entropy': hist_entropy,
        'lab_L_mean': lab_L_mean,
    }


# ---------------- UNIT TESTS ----------------

def test_categorize_small(classifier):
    feat = make_features(rel_size=0.02)
    res = classifier.classify_coin(feat)
    assert res['category'] == 'Small'

def test_categorize_medium(classifier):
    feat = make_features(rel_size=0.04)
    res = classifier.classify_coin(feat)
    assert res['category'] == 'Medium'

def test_categorize_large(classifier):
    feat = make_features(rel_size=0.06)
    res = classifier.classify_coin(feat)
    assert res['category'] == 'Large'

def test_validate_coin_success(classifier):
    feat = make_features(rel_size=0.04, sat_mean=100, hist_entropy=2.5)
    res = classifier.classify_coin(feat)
    assert res['is_coin'] is True
    assert res['confidence'] > 0.6

def test_validate_noise_low_sat(classifier):
    # Very low saturation and low entropy should fail
    feat = make_features(rel_size=0.04, sat_mean=10, hist_entropy=0.2, color_group='UNKNOWN')
    res = classifier.classify_coin(feat)
    assert res['is_coin'] is False
    assert res['confidence'] < 0.5

def test_validate_silver_coin(classifier):
    feat = make_features(rel_size=0.04, color_group='SILVER', sat_mean=20, lab_L_mean=200, hist_entropy=2.0)
    res = classifier.classify_coin(feat)
    assert res['is_coin'] is True

def test_summarize_results(classifier):
    coins = [
        {'is_coin': True, 'category': 'Small', 'confidence': 0.8},
        {'is_coin': True, 'category': 'Small', 'confidence': 0.9},
        {'is_coin': True, 'category': 'Medium', 'confidence': 0.7},
        {'is_coin': False, 'category': 'Large', 'confidence': 0.3},
    ]
    summary = classifier.summarize_results(coins)
    assert summary['total_count'] == 3
    assert summary['total_detected'] == 4
    assert summary['category_counts']['Small'] == 2
    assert summary['category_counts']['Medium'] == 1
    assert 'Large' not in summary['category_counts'] or summary['category_counts']['Large'] == 0

def test_full_run(classifier):
    fake_phase4 = {
        "features": [
            make_features(0.02), # Small coin
            make_features(0.04), # Medium coin
            make_features(0.04, sat_mean=5, hist_entropy=0.1, color_group='UNKNOWN'), # Noise
        ]
    }
    result = classifier.run(fake_phase4)
    assert len(result['classified_coins']) == 3
    assert result['summary']['total_count'] == 2
    assert result['summary']['category_counts']['Small'] == 1
    assert result['summary']['category_counts']['Medium'] == 1
