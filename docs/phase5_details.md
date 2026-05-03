# Phase 5: Classification & Counting
## Coin Counting System — Classical CV Pipeline

---

## 1. Overview

Classification is the decision-making phase. Using the feature vectors from Phase 4, a **rule-based decision tree** assigns each coin a denomination. No ML — all rules are derived from the Euro coin specification (physical diameters + metal colors).

**Goal**: For each feature vector → assign denomination (e.g., "50c"), confidence score, value in EUR, then compute totals.

---

## 2. The Classification Strategy

### 2.1 Two-Stage Decision: Color Group First, Then Size

```
Feature Vector
      │
      ▼
┌──────────────────┐
│  Stage 1:        │
│  Color Group     │──→ COPPER → {1c, 2c, 5c}
│  Classification  │──→ GOLD   → {10c, 20c, 50c}
└──────────────────┘──→ BICOLOR → {€1, €2}
                   └──→ SILVER  → {Commemorative}
                         │
                         ▼
                   ┌──────────────┐
                   │  Stage 2:    │
                   │  Size Within │
                   │  Color Group │
                   └──────────────┘
```

This 2-stage approach is more robust than using size alone (lighting changes affect both size estimation AND color) or color alone (10c and 5c have similar sizes but different colors).

### 2.2 Euro Coin Decision Tree

```
IF color_group == COPPER:
    IF size_bin in (tiny, small):
        IF rel_size < threshold_1c_2c:
            → 1c  (smallest)
        ELSE:
            → 2c  (slightly larger)
    ELSE:
        → 5c  (largest copper)

IF color_group == GOLD:
    IF size_bin == small:
        → 10c  (smallest gold)
    ELIF size_bin == medium:
        IF rel_size < threshold_10c_20c:
            → 20c
        ELSE:
            → 20c or 10c (ambiguous → lower confidence)
    ELIF size_bin in (medium, large):
        → 50c  (largest gold)

IF color_group == BICOLOR:
    IF rel_size < threshold_1e_2e:
        → €1  (smaller bicolor)
    ELSE:
        → €2  (larger bicolor)

IF color_group == SILVER:
    → UNKNOWN (confidence = 0.3)
```

---

## 3. Implementation Steps

### Step 5.1 — Denomination Lookup Table

```python
# src/phase5_classification.py
import numpy as np
from typing import List, Dict, Optional, Tuple

# Euro coin reference data
# diameter_mm: physical diameter from EU mint specification
# color_group: metal composition → color
# value_eur: monetary value in Euros
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

# Sorted by diameter within each color group (ascending)
COPPER_BY_SIZE = ['1c', '2c', '5c']    # 16.25, 18.75, 21.25 mm
GOLD_BY_SIZE   = ['10c', '20c', '50c'] # 19.75, 22.25, 24.25 mm
BICOLOR_BY_SIZE= ['1e', '2e']          # 23.25, 25.75 mm
```

### Step 5.2 — Core Classifier

```python
class CoinClassifier:
    def __init__(self, config: dict):
        self.config = config
        
        # Size decision boundary for within-group disambiguation
        # These are rel_size fractions calibrated for 1024px working images
        self.size_splits = config.get('size_splits', {
            'copper_1c_2c':  0.022,  # < → 1c, >= → 2c/5c
            'copper_2c_5c':  0.032,  # < → 2c, >= → 5c
            'gold_10c_20c':  0.030,  # < → 10c, >= → 20c/50c
            'gold_20c_50c':  0.038,  # < → 20c, >= → 50c
            'bicolor_1e_2e': 0.043,  # < → 1e, >= → 2e
        })

    def classify_coin(self, features: Dict) -> Dict:
        """
        Apply two-stage classification: color group → size within group.
        
        Returns dict with denomination, value, confidence, and reasoning.
        """
        color_group = features.get('color_group', 'UNKNOWN')
        rel_size    = features.get('rel_size', 0.0)
        size_bin    = features.get('size_bin', 'medium')
        
        denomination = 'UNKNOWN'
        confidence   = 0.0
        reasoning    = []
        
        if color_group == 'COPPER':
            denomination, confidence, reasoning = self._classify_copper(
                rel_size, size_bin, features)
        
        elif color_group == 'GOLD':
            denomination, confidence, reasoning = self._classify_gold(
                rel_size, size_bin, features)
        
        elif color_group == 'BICOLOR':
            denomination, confidence, reasoning = self._classify_bicolor(
                rel_size, size_bin, features)
        
        elif color_group == 'SILVER':
            # Rare — return low confidence unknown
            denomination, confidence = 'UNKNOWN', 0.20
            reasoning = ['Silver color group not in standard Euro set']
        
        else:
            denomination, confidence = 'UNKNOWN', 0.10
            reasoning = [f'Unrecognized color group: {color_group}']
        
        value_eur = EURO_COINS.get(denomination, {}).get('value_eur', 0.0)
        
        return {
            'coin_index':   features['coin_index'],
            'circle':       features['circle'],
            'denomination': denomination,
            'value_eur':    value_eur,
            'confidence':   confidence,
            'color_group':  color_group,
            'reasoning':    reasoning,
            'features':     features,
        }

    def _classify_copper(self, rel_size: float, 
                          size_bin: str, features: Dict) -> Tuple:
        """Classify among 1c, 2c, 5c using size within COPPER group."""
        reasoning = ['Color group: COPPER']
        
        if rel_size < self.size_splits['copper_1c_2c']:
            den, conf = '1c', 0.75
            reasoning.append(f'rel_size={rel_size:.4f} < {self.size_splits["copper_1c_2c"]} → 1c')
        elif rel_size < self.size_splits['copper_2c_5c']:
            den, conf = '2c', 0.75
            reasoning.append(f'rel_size={rel_size:.4f} in (1c,5c) range → 2c')
        else:
            den, conf = '5c', 0.80
            reasoning.append(f'rel_size={rel_size:.4f} >= {self.size_splits["copper_2c_5c"]} → 5c')
        
        # Boost confidence if saturation strongly supports copper
        if features.get('sat_mean', 0) > 100:
            conf = min(1.0, conf + 0.10)
            reasoning.append('High saturation confirms copper')
        
        return den, conf, reasoning

    def _classify_gold(self, rel_size: float, 
                        size_bin: str, features: Dict) -> Tuple:
        """Classify among 10c, 20c, 50c using size within GOLD group."""
        reasoning = ['Color group: GOLD']
        
        if rel_size < self.size_splits['gold_10c_20c']:
            den, conf = '10c', 0.75
            reasoning.append(f'rel_size={rel_size:.4f} → 10c (smallest gold)')
        elif rel_size < self.size_splits['gold_20c_50c']:
            den, conf = '20c', 0.70
            reasoning.append(f'rel_size={rel_size:.4f} → 20c')
        else:
            den, conf = '50c', 0.75
            reasoning.append(f'rel_size={rel_size:.4f} → 50c (largest gold)')
        
        # Boost if b* (yellowness) strongly supports gold
        if features.get('lab_b_mean', 128) > 148:
            conf = min(1.0, conf + 0.08)
            reasoning.append('High LAB b* confirms gold')
        
        return den, conf, reasoning

    def _classify_bicolor(self, rel_size: float, 
                           size_bin: str, features: Dict) -> Tuple:
        """Classify between €1 and €2 using size within BICOLOR group."""
        reasoning = ['Color group: BICOLOR (hue variance detected)']
        
        if rel_size < self.size_splits['bicolor_1e_2e']:
            den, conf = '1e', 0.72
            reasoning.append(f'rel_size={rel_size:.4f} → €1')
        else:
            den, conf = '2e', 0.72
            reasoning.append(f'rel_size={rel_size:.4f} → €2')
        
        # Bicolor is inherently harder to classify — cap confidence
        conf = min(conf, 0.80)
        
        return den, conf, reasoning
```

### Step 5.3 — Confidence Calibration

```python
    def calibrate_confidence(self, result: Dict, 
                              all_results: List[Dict]) -> Dict:
        """
        Adjust confidence based on contextual information:
        
        1. RELATIVE SIZE CONSISTENCY: If this coin's size relative to other
           detected coins is inconsistent with denomination pair, reduce confidence.
        
        2. BOUNDARY PROXIMITY: Partially visible coins → reduce confidence.
        
        3. AMBIGUOUS PAIRS: 1e/50c and 2e/1e have similar sizes → flag.
        """
        r = result.copy()
        x, y, radius = r['circle']
        den = r['denomination']
        
        # Check if coin is near image boundary (partially cut off)
        # This would have been flagged in Phase 2's boundary filter
        # If radius is atypically large for denomination → suspicious
        if den != 'UNKNOWN':
            expected_rel = EURO_COINS[den]['diameter_mm'] / 25.75  # Normalize to 2e
            actual_rel   = radius / max(c['circle'][2] for c in all_results)
            
            ratio_diff = abs(expected_rel - actual_rel) / expected_rel
            if ratio_diff > 0.25:
                r['confidence'] *= 0.75
                r['reasoning'].append(
                    f'Size inconsistent with {den} (diff={ratio_diff:.2%}) → reduced confidence')
        
        return r
```

### Step 5.4 — Counting & Totaling

```python
    def count_and_total(self, classified_coins: List[Dict]) -> Dict:
        """
        Count coins per denomination and compute total monetary value.
        
        Uses integer arithmetic (cents) to avoid floating-point errors:
        e.g., 0.1 + 0.2 ≠ 0.3 in float, but 10 + 20 = 30 in int.
        """
        # Count per denomination
        denomination_counts = {}
        denomination_values = {}
        
        for coin in classified_coins:
            den = coin['denomination']
            val = coin['value_eur']
            
            denomination_counts[den] = denomination_counts.get(den, 0) + 1
            denomination_values[den] = val
        
        # Compute total using integer cents to avoid floating point issues
        total_cents = 0
        for coin in classified_coins:
            val_cents = round(coin['value_eur'] * 100)
            total_cents += val_cents
        
        total_eur = total_cents / 100.0
        
        # Build breakdown table
        breakdown = []
        for den in sorted(denomination_values.keys(), 
                          key=lambda d: denomination_values.get(d, 0)):
            count = denomination_counts[den]
            unit_val = denomination_values[den]
            subtotal = round(count * unit_val * 100) / 100.0
            breakdown.append({
                'denomination': den,
                'count':        count,
                'unit_value':   unit_val,
                'subtotal_eur': subtotal,
            })
        
        # Average confidence
        confidences = [c['confidence'] for c in classified_coins 
                       if c['denomination'] != 'UNKNOWN']
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        return {
            'total_eur':           total_eur,
            'total_coins':         len(classified_coins),
            'breakdown':           breakdown,
            'denomination_counts': denomination_counts,
            'avg_confidence':      avg_confidence,
            'low_confidence_count': sum(1 for c in classified_coins 
                                         if c['confidence'] < 0.5),
        }

    def run(self, phase4_result: dict) -> dict:
        """Classify all coins and compute totals."""
        features_list = phase4_result['features']
        
        # Classify each coin
        classified = []
        for feat in features_list:
            result = self.classify_coin(feat)
            classified.append(result)
        
        # Calibrate confidence using cross-coin context
        calibrated = []
        for result in classified:
            cal = self.calibrate_confidence(result, classified)
            calibrated.append(cal)
        
        # Count and total
        summary = self.count_and_total(calibrated)
        
        return {
            'classified_coins': calibrated,
            'summary':          summary,
        }
```

---

## 4. Strategy Explanation

### 4.1 Why Rule-Based Over ML?

| Criterion | Rule-Based | ML/DL |
|---|---|---|
| Training data needed | None | Yes (labeled images) |
| Interpretable | Yes — readable rules | No — black box |
| Generalizes to new angles/lighting | With tuning | With retraining |
| Deployment size | Tiny (<1KB rules) | 10MB+ model weights |
| Assignment-appropriate | ✅ | ❌ (out of scope) |

### 4.2 Disambiguation of Hard Cases

The most difficult pairs are:
- **10c vs 2c**: Similar size, different color (gold vs copper)
- **€1 vs 50c**: Similar size, different color (bicolor vs gold)
- **1c vs 2c**: Very similar size and color

Solution: The two-stage approach (color first, then size within group) eliminates cross-group confusion. Within-group size thresholds are calibrated using physical diameter ratios.

### 4.3 Integer Arithmetic for Currency

```
✗ 0.10 + 0.20 + 0.50 = 0.8000000000000000 (floating point error possible)
✓ 10   + 20   + 50   = 80 cents → 0.80 EUR (exact)
```

Always work in integer cents internally; convert to EUR only for display.

---

## 5. `config.yaml` Parameters for Phase 5

```yaml
classification:
  size_splits:
    copper_1c_2c:  0.022
    copper_2c_5c:  0.032
    gold_10c_20c:  0.030
    gold_20c_50c:  0.038
    bicolor_1e_2e: 0.043
  low_confidence_threshold: 0.50  # Flag coins below this confidence
```

---

## 6. Expected Outputs

| Output | Description |
|---|---|
| `classified_coins` | List of result dicts with denomination, value, confidence |
| `summary.total_eur` | Total monetary value (float, 2 decimal places) |
| `summary.breakdown` | Per-denomination count and subtotal |
| `summary.avg_confidence` | Mean classification confidence |

**Expected output for a €1.91 set** (1×€1, 1×50c, 2×20c, 1×1c):
```
Denomination | Count | Unit   | Subtotal
-------------|-------|--------|----------
1c           |   1   | €0.01  | €0.01
20c          |   2   | €0.20  | €0.40
50c          |   1   | €0.50  | €0.50
€1           |   1   | €1.00  | €1.00
-------------|-------|--------|----------
TOTAL        |   5   |        | €1.91
```

---

## 7. Test Cases (Phase 5 Complete When All Pass)

```python
# tests/test_phase5_classification.py
import pytest
from src.phase5_classification import CoinClassifier, EURO_COINS

CONFIG = {
    'size_splits': {
        'copper_1c_2c':  0.022,
        'copper_2c_5c':  0.032,
        'gold_10c_20c':  0.030,
        'gold_20c_50c':  0.038,
        'bicolor_1e_2e': 0.043,
    }
}

@pytest.fixture
def classifier():
    return CoinClassifier(config=CONFIG)

def make_features(color_group, rel_size, coin_index=0, sat_mean=100, lab_b_mean=145):
    return {
        'coin_index': coin_index,
        'circle': (100, 100, 50),
        'color_group': color_group,
        'rel_size': rel_size,
        'size_bin': 'medium',
        'sat_mean': sat_mean,
        'lab_b_mean': lab_b_mean,
        'hue_mean': 20.0,
        'hue_std': 5.0,
    }

# TC-5.1: Small copper coin classified as 1c
def test_classify_1c(classifier):
    feat = make_features('COPPER', rel_size=0.018)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == '1c'
    assert result['value_eur'] == 0.01

# TC-5.2: Medium copper coin classified as 5c
def test_classify_5c(classifier):
    feat = make_features('COPPER', rel_size=0.036)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == '5c'
    assert result['value_eur'] == 0.05

# TC-5.3: Small gold coin classified as 10c
def test_classify_10c(classifier):
    feat = make_features('GOLD', rel_size=0.027)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == '10c'
    assert result['value_eur'] == 0.10

# TC-5.4: Large gold coin classified as 50c
def test_classify_50c(classifier):
    feat = make_features('GOLD', rel_size=0.045)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == '50c'
    assert result['value_eur'] == 0.50

# TC-5.5: Small bicolor classified as 1e
def test_classify_1e(classifier):
    feat = make_features('BICOLOR', rel_size=0.040)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == '1e'
    assert result['value_eur'] == 1.00

# TC-5.6: Large bicolor classified as 2e
def test_classify_2e(classifier):
    feat = make_features('BICOLOR', rel_size=0.052)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == '2e'
    assert result['value_eur'] == 2.00

# TC-5.7: Unknown color group returns UNKNOWN denomination
def test_classify_unknown_color(classifier):
    feat = make_features('SILVER', rel_size=0.035)
    result = classifier.classify_coin(feat)
    assert result['denomination'] == 'UNKNOWN'
    assert result['confidence'] < 0.5

# TC-5.8: Confidence is between 0 and 1
def test_confidence_range(classifier):
    for color, size in [('COPPER', 0.025), ('GOLD', 0.035), ('BICOLOR', 0.045)]:
        feat = make_features(color, size)
        result = classifier.classify_coin(feat)
        assert 0.0 <= result['confidence'] <= 1.0

# TC-5.9: Total value computed correctly (integer arithmetic)
def test_total_value_accuracy(classifier):
    # 3 × 10c + 1 × 50c = €0.80
    coins = [
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.8},
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.8},
        {'denomination': '10c', 'value_eur': 0.10, 'confidence': 0.8},
        {'denomination': '50c', 'value_eur': 0.50, 'confidence': 0.8},
    ]
    summary = classifier.count_and_total(coins)
    assert abs(summary['total_eur'] - 0.80) < 0.001, f"Expected 0.80, got {summary['total_eur']}"

# TC-5.10: Denomination counts are correct
def test_denomination_counts(classifier):
    coins = [
        {'denomination': '1c',  'value_eur': 0.01, 'confidence': 0.8},
        {'denomination': '1c',  'value_eur': 0.01, 'confidence': 0.8},
        {'denomination': '50c', 'value_eur': 0.50, 'confidence': 0.8},
    ]
    summary = classifier.count_and_total(coins)
    assert summary['denomination_counts']['1c']  == 2
    assert summary['denomination_counts']['50c'] == 1
    assert summary['total_coins'] == 3
```

### Phase 5 Completion Checklist

- [ ] All 10 test cases pass
- [ ] Classification accuracy ≥ 85% on ground truth test set
- [ ] Total value correct for known test images
- [ ] Confidence scores are well-calibrated (high confidence = high accuracy)
- [ ] `UNKNOWN` classification rate < 5% on clean images
- [ ] Integer arithmetic prevents floating-point rounding errors in totals
