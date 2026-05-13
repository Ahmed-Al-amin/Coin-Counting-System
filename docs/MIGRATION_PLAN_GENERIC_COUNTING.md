# Migration Plan: Currency-Specific → Generic Coin Counting
## Coin Counting System · Generic Detection & Counting

This document is a step-by-step plan for generalizing the project from
the Euro-specific version to a generic system that detects and counts
all coins in an image regardless of their currency or denomination.

It covers every file that must be changed or cleaned up, in the exact
order you should do the work, to remove currency awareness while
maintaining the 6-phase modular structure.

---

## Overview: What Changes and What Does Not

The goal is to move from "How much money is here?" to "How many coins are here?".

**Files you DELETE (1 file):**
- `src/coin_registry.py` (if created) — this abstraction is no longer needed.

**Files you MODIFY (4 files):**
- `src/classification.py` (Phase 5)
- `src/output.py` (Phase 6)
- `config.yaml`
- `main.py`

**Files you DO NOT TOUCH at all:**
- `src/preprocessing.py` (Phase 1)
- `src/detection.py` (Phase 2)
- `src/segmentation.py` (Phase 3)
- `src/features.py` (Phase 4) — these remain useful for generic validation.

---

## Step 1 — Clean Up Architecture

The generic system does not need a database of diameters or values. 

1. **Delete `src/coin_registry.py`**: If you previously created this file following the multi-currency plan, delete it.
2. **Remove Imports**: Ensure no files in `src/` attempt to import from `coin_registry.py`.

---

## Step 2 — Modify `src/classification.py` (Phase 5)

Phase 5 transitions from a "Denomination Classifier" to a "Coin Validator and Categorizer".

### What to remove
- Remove `EURO_COINS` or any currency-specific dictionaries.
- Remove logic that maps "color + size" to a specific cent or euro value.
- Remove `total_eur` from the summary output.

### What to add
- **Generic Categorization**: Group coins by size labels: `Small`, `Medium`, `Large` based on relative size thresholds.
- **Validation Logic**: Use the features from Phase 4 (like color saturation or hue entropy) to confirm if a detected circle is likely a coin or just background noise (False Positive).
- **Generic Summary**: The `summary` dictionary should now focus on `total_count` and counts per size category (e.g., `small_count`, `medium_count`, `large_count`).

---

## Step 3 — Modify `src/output.py` (Phase 6)

The visual and data output should focus on counts rather than monetary values.

### What to change in `print_table`
- Remove the "Value", "Unit", and "Subtotal" columns.
- Add a "Category" (Small/Medium/Large) and "Confidence" column.
- The final row should show "TOTAL COINS" instead of "TOTAL VALUE".

### What to change in `export_json`
- Remove `value_eur` and `denomination` from the per-coin entries.
- Add `category` (size-based) and `is_coin` (boolean validation).
- Update the summary section to show `total_count`.

### What to change in `draw_annotations`
- Change labels above coins from "50c" or "€1" to generic "Coin #1", "Coin #2", etc., or just their size category "Medium".

---

## Step 4 — Modify `config.yaml`

Remove all currency-specific parameters to simplify the configuration.

### What to remove
- Remove the `size_splits` block that references denominations like `copper_1c_2c`.
- Remove any `currencies` list.

### What to add
- **Generic Size Splits**: Add thresholds for generic categories:
  ```yaml
  classification:
    size_thresholds:
      small_medium: 0.035
      medium_large: 0.055
    low_confidence_threshold: 0.50
  ```

---

## Step 5 — Modify `main.py`

Update the entry point to reflect the shift in purpose and fix the final logs.

### What to change
- Update `DEFAULT_CONFIG` to use generic size thresholds.
- Change the final log message from:
  `logger.info(f"Total value: €{phase5_result['summary']['total_eur']:.2f}")`
  to:
  `logger.info(f"Total coins detected: {phase5_result['summary']['total_count']}")`

---

## Step 6 — Verify the migration

1. **Check Pipeline**: Run `python main.py --image data/sample_images/mixed_coins.jpg`.
2. **Verify Output**: Ensure the console table shows a count of coins without any Euro symbols.
3. **Verify JSON**: Check that `results.json` contains no currency-related fields.
4. **Tests**: Update `tests/test_phase5_classification.py` to assert counts and generic categories instead of monetary values.

---

## New Project Structure (Generic)

```
coin_counter/
│
├── main.py                              [CHANGED] Generic logging
├── config.yaml                          [CHANGED] Generic thresholds
│
├── src/
│   ├── preprocessing.py                 [SAME]
│   ├── detection.py                     [SAME]
│   ├── segmentation.py                  [SAME]
│   ├── features.py                      [SAME]
│   ├── classification.py                [CHANGED] Generic validator
│   ├── output.py                        [CHANGED] Count-based report
│   └── utils/ ...                       [SAME]
```
