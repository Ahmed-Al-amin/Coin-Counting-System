# Phase 5: Validation & Categorization
## Generic Coin Counting System — Classical CV Pipeline

---

## 1. Overview

Validation and Categorization is the phase where detected circles are confirmed as coins and grouped by size. Using the feature vectors from Phase 4, a **heuristic validator** determines if a candidate circle is likely a coin or background noise, and then assigns a size category (Small, Medium, or Large).

**Goal**: For each feature vector → assign status (Coin/Noise), size category, and confidence score, then compute totals.

---

## 2. The Generic Strategy

### 2.1 Multi-Feature Validation

Instead of hardcoded denominations, we use heuristics to confirm "coin-ness":

- **Saturation**: Coins (copper, gold, even silver) usually contrast with flat backgrounds.
- **Surface Texture (Entropy)**: Real coins have stamped designs that create a specific range of hue/intensity entropy. Uniform circles (like holes or dots) are filtered out.
- **Color Groups**: Matches to known metallic groups (Copper, Gold, Silver, Bicolor) boost confidence.

### 2.2 Size Categorization

Coins are grouped based on their relative radius ($r_{rel} = \frac{r}{diagonal}$):

```
IF rel_size < threshold_small_medium:
    → Small
ELIF rel_size < threshold_medium_large:
    → Medium
ELSE:
    → Large
```

---

## 3. Implementation Steps

### Step 5.1 — Validator Logic

```python
def _validate_coin(features: Dict) -> Tuple[bool, float, List[str]]:
    """
    Score confidence based on:
    1. Saturation (Coins > 40)
    2. Hue Entropy (Typical coins: 1.0 - 4.0)
    3. Known color groups
    """
```

### Step 5.2 — Size Categorization

```python
# config.yaml
classification:
  size_thresholds:
    small_medium: 0.035
    medium_large: 0.055
```

---

## 4. Expected Outputs

| Output | Description |
|---|---|
| `classified_coins` | List of result dicts with category, is_coin, confidence |
| `summary.total_count` | Total number of validated coins |
| `summary.breakdown` | Count per size category |
| `summary.avg_confidence` | Mean validation confidence |

---

## 5. Phase 5 Completion Checklist

- [ ] Validator effectively filters non-coin circles (Noise)
- [ ] Categorization correctly labels Small/Medium/Large based on config
- [ ] Confidence scores are well-calibrated
- [ ] Summary counts match manual ground truth
