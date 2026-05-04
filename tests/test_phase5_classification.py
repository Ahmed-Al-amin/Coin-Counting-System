import pytest
import numpy as np
from src.classification import CoinClassifier, EURO_COINS


# ---------------- CONFIG ----------------
CONFIG = {
    'size_splits': {
        'copper_1c_2c':  0.022,
        'copper_2c_5c':  0.032,
        'gold_10c_20c':  0.030,
        'gold_20c_50c':  0.038,
        'bicolor_1e_2e': 0.043,
    }
}


# ---------------- FIXTURE ----------------
@pytest.fixture
def classifier():
    return CoinClassifier(CONFIG)


# ---------------- HELPERS ----------------
def make_features(color_group, rel_size, coin_index=0,
                  sat_mean=100, lab_b_mean=145):
    return {
        'coin_index': coin_index,
        'circle': (100, 100, 50),  # (x, y, radius)
        'color_group': color_group,
        'rel_size': rel_size,
        'size_bin': 'medium',
        'sat_mean': sat_mean,
        'lab_b_mean': lab_b_mean,
    }


# ---------------- UNIT TESTS ----------------

def test_classify_1c(classifier):
    feat = make_features('COPPER', 0.018)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '1c'
    assert res['value_eur'] == 0.01


def test_classify_2c(classifier):
    feat = make_features('COPPER', 0.023)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '2c'
    assert res['value_eur'] == 0.02


def test_classify_5c(classifier):
    feat = make_features('COPPER', 0.036)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '5c'
    assert res['value_eur'] == 0.05


def test_classify_10c(classifier):
    feat = make_features('GOLD', 0.028)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '10c'
    assert res['value_eur'] == 0.10


def test_classify_20c(classifier):
    feat = make_features('GOLD', 0.033)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '20c'
    assert res['value_eur'] == 0.20


def test_classify_50c(classifier):
    feat = make_features('GOLD', 0.045)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '50c'
    assert res['value_eur'] == 0.50


def test_classify_1e(classifier):
    feat = make_features('BICOLOR', 0.040)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '1e'
    assert res['value_eur'] == 1.00


def test_classify_2e(classifier):
    feat = make_features('BICOLOR', 0.050)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == '2e'
    assert res['value_eur'] == 2.00


def test_unknown_color(classifier):
    feat = make_features('SILVER', 0.03)
    res = classifier.classify_coin(feat)

    assert res['denomination'] == 'UNKNOWN'
    assert res['confidence'] < 0.5


def test_confidence_bounds(classifier):
    feat = make_features('COPPER', 0.03)
    res = classifier.classify_coin(feat)

    assert 0.0 <= res['confidence'] <= 1.0


# ---------------- TOTAL TESTS ----------------

def test_total_calculation(classifier):
    coins = [
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.8},
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.8},
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.8},
        {'denomination': '50c', 'value_eur': 0.50, 'confidence': 0.8},
    ]

    summary = classifier.count_and_total(coins)

    assert summary['total_coins'] == 4
    assert abs(summary['total_eur'] - 0.80) < 0.001


def test_denomination_counts(classifier):
    coins = [
        {'denomination': '1c', 'value_eur': 0.01, 'confidence': 0.8},
        {'denomination': '1c', 'value_eur': 0.01, 'confidence': 0.8},
        {'denomination': '50c', 'value_eur': 0.50, 'confidence': 0.8},
    ]

    summary = classifier.count_and_total(coins)

    assert summary['denomination_counts']['1c'] == 2
    assert summary['denomination_counts']['50c'] == 1
    assert summary['total_coins'] == 3


def test_avg_confidence(classifier):
    coins = [
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.6},
        {'denomination': '20c', 'value_eur': 0.20, 'confidence': 0.8},
    ]

    summary = classifier.count_and_total(coins)

    assert 0.6 <= summary['avg_confidence'] <= 0.8


# ---------------- RUN PIPELINE TEST ----------------

def test_full_pipeline(classifier):
    fake_phase4 = {
        "features": [
            make_features("COPPER", 0.018),
            make_features("GOLD", 0.033),
            make_features("BICOLOR", 0.050),
        ]
    }

    result = classifier.run(fake_phase4)

    assert "classified_coins" in result
    assert "summary" in result
    assert len(result["classified_coins"]) == 3
    assert result["summary"]["total_coins"] == 3