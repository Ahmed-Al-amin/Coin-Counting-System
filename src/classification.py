import numpy as np
from typing import List, Dict, Tuple


EURO_COINS = {
    '1c':  {'value_eur': 0.01, 'diameter_mm': 16.25, 'color_group': 'COPPER'},
    '2c':  {'value_eur': 0.02, 'diameter_mm': 18.75, 'color_group': 'COPPER'},
    '5c':  {'value_eur': 0.05, 'diameter_mm': 21.25, 'color_group': 'COPPER'},
    '10c': {'value_eur': 0.10, 'diameter_mm': 19.75, 'color_group': 'GOLD'},
    '20c': {'value_eur': 0.20, 'diameter_mm': 22.25, 'color_group': 'GOLD'},
    '50c': {'value_eur': 0.50, 'diameter_mm': 24.25, 'color_group': 'GOLD'},
    '1e':  {'value_eur': 1.00, 'diameter_mm': 23.25, 'color_group': 'BICOLOR'},
    '2e':  {'value_eur': 2.00, 'diameter_mm': 25.75, 'color_group': 'BICOLOR'},
}


class CoinClassifier:
    def __init__(self, config: dict):
        self.config = config

        self.size_splits = config.get('size_splits', {
            'copper_1c_2c':  0.022,
            'copper_2c_5c':  0.032,
            'gold_10c_20c':  0.030,
            'gold_20c_50c':  0.038,
            'bicolor_1e_2e': 0.043,
        })

    # ---------------- MAIN CLASSIFIER ----------------
    def classify_coin(self, features: Dict) -> Dict:
        color_group = features.get('color_group', 'UNKNOWN')
        rel_size = features.get('rel_size', 0.0)
        size_bin = features.get('size_bin', 'medium')

        denomination = 'UNKNOWN'
        confidence = 0.0
        reasoning = []

        if color_group == 'COPPER':
            denomination, confidence, reasoning = self._classify_copper(rel_size, size_bin, features)

        elif color_group == 'GOLD':
            denomination, confidence, reasoning = self._classify_gold(rel_size, size_bin, features)

        elif color_group == 'BICOLOR':
            denomination, confidence, reasoning = self._classify_bicolor(rel_size, size_bin, features)

        elif color_group == 'SILVER':
            denomination, confidence = 'UNKNOWN', 0.2
            reasoning = ['Silver not standard Euro coin']

        else:
            denomination, confidence = 'UNKNOWN', 0.1
            reasoning = ['Unknown color group']

        value_eur = EURO_COINS.get(denomination, {}).get('value_eur', 0.0)

        return {
            'coin_index': features['coin_index'],
            'circle': features['circle'],
            'denomination': denomination,
            'value_eur': value_eur,
            'confidence': confidence,
            'color_group': color_group,
            'reasoning': reasoning,
        }

    # ---------------- COPPER ----------------
    def _classify_copper(self, rel_size: float, size_bin: str, features: Dict) -> Tuple:
        reasoning = ['COPPER group']

        if rel_size < self.size_splits['copper_1c_2c']:
            den, conf = '1c', 0.75
        elif rel_size < self.size_splits['copper_2c_5c']:
            den, conf = '2c', 0.75
        else:
            den, conf = '5c', 0.80

        if features.get('sat_mean', 0) > 100:
            conf = min(1.0, conf + 0.1)

        return den, conf, reasoning

    # ---------------- GOLD ----------------
    def _classify_gold(self, rel_size: float, size_bin: str, features: Dict) -> Tuple:
        reasoning = ['GOLD group']

        if rel_size < self.size_splits['gold_10c_20c']:
            den, conf = '10c', 0.75
        elif rel_size < self.size_splits['gold_20c_50c']:
            den, conf = '20c', 0.70
        else:
            den, conf = '50c', 0.75

        if features.get('lab_b_mean', 128) > 148:
            conf = min(1.0, conf + 0.08)

        return den, conf, reasoning

    # ---------------- BICOLOR ----------------
    def _classify_bicolor(self, rel_size: float, size_bin: str, features: Dict) -> Tuple:
        reasoning = ['BICOLOR group']

        if rel_size < self.size_splits['bicolor_1e_2e']:
            den, conf = '1e', 0.72
        else:
            den, conf = '2e', 0.72

        return den, min(conf, 0.80), reasoning

    # ---------------- CONFIDENCE CALIBRATION ----------------
    def calibrate_confidence(self, result: Dict, all_results: List[Dict]) -> Dict:
        r = result.copy()

        if r['denomination'] == 'UNKNOWN':
            return r

        max_radius = max(c['circle'][2] for c in all_results)
        actual_rel = r['circle'][2] / max_radius

        expected_rel = EURO_COINS[r['denomination']]['diameter_mm'] / 25.75

        diff = abs(expected_rel - actual_rel) / expected_rel

        if diff > 0.25:
            r['confidence'] *= 0.75
            r['reasoning'].append('Size inconsistency detected')

        return r

    # ---------------- TOTAL ----------------
    def count_and_total(self, coins: List[Dict]) -> Dict:
        denomination_counts = {}
        denomination_values = {}

        total_cents = 0

        for coin in coins:
            den = coin['denomination']
            val = coin['value_eur']

            denomination_counts[den] = denomination_counts.get(den, 0) + 1
            denomination_values[den] = val

            total_cents += round(val * 100)

        total_eur = total_cents / 100.0

        breakdown = []
        for den in sorted(denomination_values.keys(), key=lambda d: denomination_values[d]):
            count = denomination_counts[den]
            unit = denomination_values[den]

            breakdown.append({
                'denomination': den,
                'count': count,
                'unit_value': unit,
                'subtotal_eur': round(count * unit, 2)
            })

        confidences = [c['confidence'] for c in coins if c['denomination'] != 'UNKNOWN']
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            'total_eur': total_eur,
            'total_coins': len(coins),
            'breakdown': breakdown,
            'denomination_counts': denomination_counts,
            'avg_confidence': avg_conf,
            'low_confidence_count': sum(1 for c in coins if c['confidence'] < 0.5),
        }

    # ---------------- RUN ----------------
    def run(self, phase4_result: dict) -> dict:
        features_list = phase4_result['features']

        classified = [self.classify_coin(f) for f in features_list]

        calibrated = [self.calibrate_confidence(c, classified) for c in classified]

        summary = self.count_and_total(calibrated)

        return {
            'classified_coins': calibrated,
            'summary': summary
        }