# Migration Plan: Multi-Currency & Generic Counting Hybrid
## Coin Counting System · Choice-Based Pipeline

This document is a step-by-step plan for upgrading the project to a 
hybrid system. It allows the user to choose between:
1. **Multi-Currency Mode**: Detects EUR, USD, or EGP with monetary totals.
2. **Generic Mode**: Detects and counts coins by size without currency awareness.

The choice is controlled via `config.yaml`.

---

## Overview: What Changes and What Does Not

**Files you CREATE from scratch (1 file):**
- `src/coin_registry.py` (Central metadata for all currencies)

**Files you MODIFY (4 files):**
- `src/classification.py` (Phase 5: Dual-mode logic)
- `src/output.py` (Phase 6: Dynamic reporting)
- `config.yaml` (Mode & Currency toggles)
- `main.py` (Bootstrap mode selection)

**Files you DO NOT TOUCH at all:**
- `src/preprocessing.py`, `src/detection.py`, `src/segmentation.py`, `src/features.py`
- These phases remain currency-agnostic and work for both modes.

---

## Step 1 — Create `src/coin_registry.py`

This file serves as the database for the **Multi-Currency Mode**.

### What to include:
- A `COIN_DB` dictionary containing specs for `EUR`, `USD`, and `EGP`.
- Fields: `value`, `diameter_mm`, `color_group`, `symbol`, `currency_symbol`.
- Helper functions: `get_currency_coins()`, `get_all_active_coins()`, `supported_currencies()`.

---

## Step 2 — Modify `src/classification.py` (Phase 5)

Phase 5 becomes a **Switchable Classifier**.

### Implementation Strategy:
- **`__init__`**: Read `mode` (default: "generic") and `currencies` (default: ["EUR"]) from config.
- **`run()`**: 
  - If `mode == "currency"`: Execute currency-specific Stage 0 (currency detection) and Stage 1 (denomination mapping).
  - If `mode == "generic"`: Execute generic size-labeling (Small, Medium, Large).
- **`count_and_total()`**:
  - Currency mode: Accumulate totals per currency (e.g., `totals_by_currency`).
  - Generic mode: Accumulate `total_count` and per-size counts.

---

## Step 3 — Modify `src/output.py` (Phase 6)

The output module must now handle two completely different visual styles.

### Logic Changes:
- **`print_table()`**: 
  - If "currency": Print currency headers, unit values, and monetary subtotals.
  - If "generic": Print simple "Coin ID | Category | Confidence" table.
- **`export_json()`**:
  - Dynamically include/exclude fields like `value_eur` or `category` based on the active mode.
- **`draw_annotations()`**:
  - Label coins with their currency symbol (e.g., "€1") or generic category (e.g., "Medium") depending on mode.

---

## Step 4 — Modify `config.yaml`

This is where the user makes their choice.

```yaml
classification:
  mode: "currency"         # Options: "currency" or "generic"
  currencies: ["EUR", "USD"] # Only used if mode is "currency"
  
  # Used if mode is "generic"
  size_thresholds:
    small_medium: 0.035
    medium_large: 0.055
    
  # Used if mode is "currency" (overrides defaults)
  size_splits:
    EUR:
      COPPER_0_1: 0.022
```

---

## Step 5 — Modify `main.py`

Update the entry point to handle the dual-mode summary.

### Changes:
- Update `DEFAULT_CONFIG` to include `mode: "generic"`.
- Update the final logger message:
  ```python
  if phase5_result['mode'] == 'currency':
      logger.info(f"Total value: {phase5_result['summary']['formatted_grand_total']}")
  else:
      logger.info(f"Total coins detected: {phase5_result['summary']['total_count']}")
  ```

---

## Step 6 — Verification

1. **Test Generic**: Set `mode: "generic"` in config and verify no currency symbols appear.
2. **Test Currency**: Set `mode: "currency"` and `currencies: ["USD"]` and verify dollar amounts.
3. **Test Hybrid**: Set `mode: "currency"` and `currencies: ["EUR", "EGP"]` to verify multi-currency handling.

---

## New Project Structure (Hybrid)

```
coin_counter/
│
├── main.py                              [CHANGED]
├── config.yaml                          [CHANGED] Added 'mode' toggle
│
├── src/
│   ├── coin_registry.py                 [NEW] Multi-currency DB
│   ├── classification.py                [CHANGED] Switchable Logic
│   ├── output.py                        [CHANGED] Dynamic Reporting
│   └── ... (others same)
```
