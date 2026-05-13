import numpy as np
from typing import List, Dict, Tuple


class CoinClassifier:
    def __init__(self, config: dict):
        self.config = config

        # Generic size thresholds for categorization
        self.size_thresholds = config.get('size_thresholds', {
            'small_medium': 0.035,
            'medium_large': 0.055,
        })
        self.low_confidence_threshold = config.get('low_confidence_threshold', 0.50)

    # ---------------- MAIN CLASSIFIER ----------------
    def classify_coin(self, features: Dict) -> Dict:
        """
        Categorize a detected circle by size and validate if it's likely a coin.
        """
        rel_size = features.get('rel_size', 0.0)
        
        # Categorize by size
        if rel_size < self.size_thresholds['small_medium']:
            category = 'Small'
        elif rel_size < self.size_thresholds['medium_large']:
            category = 'Medium'
        else:
            category = 'Large'

        # Validation Logic: Check if it's actually a coin based on features
        is_coin, confidence, reasoning = self._validate_coin(features)

        return {
            'coin_index': features['coin_index'],
            'circle': features['circle'],
            'category': category,
            'is_coin': is_coin,
            'confidence': confidence,
            'color_group': features.get('color_group', 'UNKNOWN'),
            'reasoning': reasoning,
        }

    def _validate_coin(self, features: Dict) -> Tuple[bool, float, List[str]]:
        """
        Determine if a detected circle is a coin or noise/background.
        Uses saturation, entropy, and color group to score confidence.
        """
        reasoning = []
        confidence = 0.5  # Base confidence
        
        sat_mean = features.get('sat_mean', 0)
        hue_entropy = features.get('hist_entropy', 0)
        color_group = features.get('color_group', 'UNKNOWN')

        # Heuristic 1: Coins usually have some saturation unless silver/worn
        if sat_mean > 40:
            confidence += 0.1
            reasoning.append("Good color saturation")
        elif color_group == 'SILVER' and features.get('lab_L_mean', 0) > 150:
            confidence += 0.1
            reasoning.append("High-luminance silver candidate")
        else:
            confidence -= 0.1
            reasoning.append("Low saturation/contrast")

        # Heuristic 2: Coins have surface texture (entropy)
        if 1.0 < hue_entropy < 4.0:
            confidence += 0.15
            reasoning.append("Typical coin surface texture")
        elif hue_entropy < 0.5:
            confidence -= 0.2
            reasoning.append("Surface too uniform (likely noise)")

        # Heuristic 3: Known color groups
        if color_group != 'UNKNOWN':
            confidence += 0.1
            reasoning.append(f"Matches {color_group} group")
        else:
            confidence -= 0.15
            reasoning.append("Color group unknown")

        # Clamp confidence
        confidence = float(np.clip(confidence, 0.0, 1.0))
        is_coin = confidence >= self.low_confidence_threshold

        return is_coin, confidence, reasoning

    # ---------------- SUMMARY ----------------
    def summarize_results(self, coins: List[Dict]) -> Dict:
        """
        Generate a summary focusing on total counts and size breakdown.
        """
        valid_coins = [c for c in coins if c['is_coin']]
        
        category_counts = {}
        for coin in valid_coins:
            cat = coin['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1

        breakdown = []
        for cat in ['Small', 'Medium', 'Large']:
            count = category_counts.get(cat, 0)
            breakdown.append({
                'category': cat,
                'count': count
            })

        confidences = [c['confidence'] for c in valid_coins]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        return {
            'total_count': len(valid_coins),
            'total_detected': len(coins),
            'breakdown': breakdown,
            'category_counts': category_counts,
            'avg_confidence': avg_conf,
            'low_confidence_count': sum(1 for c in coins if c['confidence'] < self.low_confidence_threshold),
        }

    # ---------------- RUN ----------------
    def run(self, phase4_result: dict) -> dict:
        features_list = phase4_result['features']

        classified = [self.classify_coin(f) for f in features_list]
        summary = self.summarize_results(classified)

        return {
            'classified_coins': classified,
            'summary': summary
        }