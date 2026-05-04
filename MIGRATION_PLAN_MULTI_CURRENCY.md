# Migration Plan: Euro-Only → Multi-Currency
## Coin Counting System · EUR + USD + EGP

This document is a step-by-step plan for upgrading the project from
the Euro-only version to one that supports Egyptian Pound, US Dollar,
and Euro — individually or all at once in a mixed image.

It covers every file that must be created or changed, in the exact
order you should do the work, with the precise additions and removals
written out so you can follow along without opening the code.

---

## Overview: What Changes and What Does Not

Before starting, it helps to know exactly how much work is involved.

**Files you CREATE from scratch (1 file):**
- `src/coin_registry.py`

**Files you MODIFY (4 files):**
- `src/phase5_classification.py`
- `src/phase6_output.py`
- `config.yaml`
- `main.py`

**Files you DO NOT TOUCH at all:**
- `src/phase1_preprocessing.py` — image processing only, no currency awareness
- `src/phase2_detection.py` — circle detection only, no currency awareness
- `src/phase3_segmentation.py` — coin cropping only, no currency awareness
- `src/phase4_features.py` — colour and size features, works for all three currencies as-is
- `src/utils/image_io.py` — file utilities, no currency awareness
- `src/utils/logger.py` — logging setup, no currency awareness
- All test files — the framework tests still pass; you add new tests for new currencies

---

## Step 1 — Create `src/coin_registry.py`

This is the only entirely new file. It does not exist in the Euro-only
version. You create it inside the `src/` folder next to the phase files.

### What this file is

A single Python file that holds one dictionary called `COIN_DB`.
That dictionary contains the physical specification of every coin in
every supported currency. Every other file in the project that needs
to know anything about a coin reads from this dictionary instead of
having the data hard-coded locally.

The file also provides six small helper functions that other modules
call to look up specific pieces of information without having to
navigate the dictionary themselves.

### The structure of `COIN_DB`

The dictionary has one key per currency, using the standard three-letter
currency code. Inside each currency is one key per denomination, and
inside each denomination is a small dictionary of five fields.

The five fields every denomination must have:

- `value` — the monetary value as a decimal number in the currency's base unit. For Euro this is euros, so 50 cents is written as 0.50. For Egyptian Pound this is pounds, so 50 piastres is written as 0.50.
- `diameter_mm` — the physical outer diameter of the coin in millimetres, taken from the official mint specification.
- `color_group` — one of four labels that describes the metal surface colour. The four valid labels are COPPER, GOLD, SILVER, and BICOLOR.
- `symbol` — the short text label that appears on the annotated image and in the HTML report, for example "25¢" or "LE1".
- `currency_symbol` — the currency prefix character used in monetary displays, for example "€", "$", or "LE".

An optional sixth field called `notes` can hold a plain-text explanation
of anything unusual about that coin, such as the fact that the US dime
is physically smaller than the nickel.

### The colour group rules

When you assign a `color_group` to a denomination, use this guide:

- COPPER — the coin surface is orange-brown. This is copper-plated steel or copper-plated zinc. Examples: all Euro cent coins below 10 cents, the US penny.
- GOLD — the coin surface is yellow-gold. This is Nordic Gold alloy or brass. Examples: Euro 10, 20, and 50 cent coins.
- SILVER — the coin surface is grey or white. This is cupronickel, nickel-plated steel, or stainless steel. Examples: all US coins except the penny and dollar, all older Egyptian coins.
- BICOLOR — the coin has two visually distinct colour zones, one ring and one centre, in different metals. Examples: Euro 1 and 2 euro, US dollar coin, newer Egyptian 50 piastre and 1 pound.

### Euro entries

```
"EUR": {
    "1c":  value=0.01, diameter=16.25mm, color_group=COPPER,  symbol="1¢",  currency_symbol="€"
    "2c":  value=0.02, diameter=18.75mm, color_group=COPPER,  symbol="2¢",  currency_symbol="€"
    "5c":  value=0.05, diameter=21.25mm, color_group=COPPER,  symbol="5¢",  currency_symbol="€"
    "10c": value=0.10, diameter=19.75mm, color_group=GOLD,    symbol="10¢", currency_symbol="€"
    "20c": value=0.20, diameter=22.25mm, color_group=GOLD,    symbol="20¢", currency_symbol="€"
    "50c": value=0.50, diameter=24.25mm, color_group=GOLD,    symbol="50¢", currency_symbol="€"
    "1e":  value=1.00, diameter=23.25mm, color_group=BICOLOR, symbol="E1",  currency_symbol="€"
    "2e":  value=2.00, diameter=25.75mm, color_group=BICOLOR, symbol="E2",  currency_symbol="€"
}
```

### US Dollar entries

```
"USD": {
    "1c":  value=0.01, diameter=19.05mm, color_group=COPPER,  symbol="1c",  currency_symbol="$"
    "5c":  value=0.05, diameter=21.21mm, color_group=SILVER,  symbol="5c",  currency_symbol="$"
    "10c": value=0.10, diameter=17.91mm, color_group=SILVER,  symbol="10c", currency_symbol="$"
    "25c": value=0.25, diameter=24.26mm, color_group=SILVER,  symbol="25c", currency_symbol="$"
    "50c": value=0.50, diameter=30.61mm, color_group=SILVER,  symbol="50c", currency_symbol="$"
    "1d":  value=1.00, diameter=26.50mm, color_group=BICOLOR, symbol="$1",  currency_symbol="$"
}
```

Note on USD: the 10-cent dime has a smaller diameter than the 5-cent
nickel. This is unusual and the classifier handles it automatically.
Do not reorder these entries — the `coin_registry.py` sorts them
internally by diameter when the classifier needs them.

### Egyptian Pound entries

```
"EGP": {
    "25p":  value=0.25, diameter=18.00mm, color_group=SILVER,  symbol="25p",  currency_symbol="LE"
    "50p":  value=0.50, diameter=20.00mm, color_group=SILVER,  symbol="50p",  currency_symbol="LE"
    "50pa": value=0.50, diameter=21.00mm, color_group=BICOLOR, symbol="50p*", currency_symbol="LE"
    "1L":   value=1.00, diameter=23.00mm, color_group=BICOLOR, symbol="LE1",  currency_symbol="LE"
    "1La":  value=1.00, diameter=20.00mm, color_group=SILVER,  symbol="LE1*", currency_symbol="LE"
}
```

Note on EGP: `50p` and `50pa` are two different 50-piastre coins that
circulate at the same time. `50p` is the older silver series, `50pa` is
the newer bicolour series. Similarly `1L` is the newer bicolour pound
and `1La` is the older silver pound. Both are included because both
circulate and both may appear in a photograph.

### The six helper functions

After the `COIN_DB` dictionary, you write six small functions. Other
files call these instead of accessing the dictionary directly.

**`get_currency_coins(currency)`** — takes a currency code like "EUR"
and returns the inner dictionary of all denominations for that currency.
Raises a clear error if the currency code is not recognised.

**`get_all_active_coins(currencies)`** — takes a list of currency codes
and merges all their denominations into one flat dictionary. Each key is
prefixed with the currency code and a colon, so "EUR:1c" and "USD:1c"
are different entries.

**`get_color_groups_for_currency(currency)`** — takes a currency code
and returns a dictionary where each key is a colour group label and the
value is a list of denomination keys for that group, sorted ascending
by physical diameter. This is what the classifier uses to know which
denominations exist in each colour group and in what size order.

**`supported_currencies()`** — returns a sorted list of all currency
codes in `COIN_DB`. The classifier calls this to validate user input.

**`largest_diameter(currency)`** — returns the diameter in millimetres
of the largest coin in a given currency. The classifier uses this to
compute relative size ratios during cross-coin confidence calibration.

**`currency_symbol(currency)`** — returns the currency symbol string
for a given currency code. The output module calls this when building
the console table and HTML report.

---

## Step 2 — Modify `src/phase5_classification.py`

This is the most significant change in the migration. The file goes from
a two-stage Euro-specific classifier to a three-stage generic classifier.

### What to remove from the Euro-only version

At the top of the file there is a dictionary called `EURO_COINS`. Remove
the entire dictionary and the constants below it (`COPPER_BY_SIZE`,
`GOLD_BY_SIZE`, `BICOLOR_BY_SIZE`).

The Euro-only version imports nothing except numpy and the logger. Remove
those bare imports and replace them with imports from `coin_registry.py`.

The three private methods `_classify_copper`, `_classify_gold`, and
`_classify_bicolor` are Euro-specific. Remove all three of them.

Inside the `__init__` method, remove the `size_splits` dictionary
that uses Euro denomination names as keys (keys like `copper_1c_2c`,
`gold_10c_20c`, `bicolor_1e_2e`). These become per-currency and use
generic index-based names.

### What to add to the class

**New imports at the top of the file:**
Add an import block that pulls in the six helper functions from
`coin_registry.py`: `COIN_DB`, `get_currency_coins`,
`get_color_groups_for_currency`, `largest_diameter`, and
`currency_symbol`.

**New attribute in `__init__`:**
Add a `currencies` attribute that reads from `config.get("currencies", ["EUR"])`.
The default of `["EUR"]` makes the new version backward-compatible with
any code that was already passing a config without a currencies key.

Add a validation loop that checks each currency code in the list against
`COIN_DB` and raises a clear error if an unrecognised code is given.

Add a `_groups` attribute that pre-computes the colour-group-to-denomination
mapping for each active currency by calling `get_color_groups_for_currency`.

**New private method `_populate_default_splits`:**
This method fills in size-split thresholds for any currency that was not
explicitly configured. The thresholds are now nested per currency, and the
keys use generic names based on colour group and position index rather than
denomination names.

The default thresholds to pre-load:

For EUR: the key `COPPER_0_1` holds the boundary between the first and
second denomination in the COPPER group when sorted by diameter, which
is the 1c/2c boundary, value 0.022. The key `COPPER_1_2` is the 2c/5c
boundary, value 0.028. The key `GOLD_0_1` is the 10c/20c boundary,
value 0.027. The key `GOLD_1_2` is the 20c/50c boundary, value 0.030.
The key `BICOLOR_0_1` is the 1e/2e boundary, value 0.038.

For USD: the key `SILVER_0_1` is the 10c/5c boundary, value 0.025.
The key `SILVER_1_2` is the 5c/25c boundary, value 0.030. The key
`SILVER_2_3` is the 25c/50c boundary, value 0.036. There is no COPPER
split because USD only has one copper coin (the penny).

For EGP: the key `SILVER_0_1` is the 25p/50p boundary, value 0.025.
The key `BICOLOR_0_1` is the 50pa/1L boundary, value 0.028.

**New private method `_detect_currency` (Stage 0):**
This method scores each active currency and returns the one that best
matches the coin's feature vector, along with a confidence factor.

The scoring works as follows. For each active currency, compute a colour
score: if the coin's colour group exists in that currency's denominations
give a score of 1.0, otherwise 0.0. Then compute a size score: convert
the expected diameter range for that currency into relative-size units
and check whether the coin's relative size falls within that range. The
combined score is the colour score multiplied by 0.6 plus the size score
multiplied by 0.4.

The currency with the highest score wins. The confidence factor is derived
from how large the gap is between the winner and the runner-up: a large
gap means the classifier is certain, a small gap means the currencies look
similar for this coin.

If only one currency is active this method immediately returns that currency
with a confidence factor of 1.0, making it a zero-cost no-op for the common
single-currency case.

**New private method `_classify_within_group` (replaces the three Euro methods):**
This single generic method replaces `_classify_copper`, `_classify_gold`,
and `_classify_bicolor`. It receives a currency code and a colour group
label, looks up the sorted list of denominations for that group from
`_groups`, then walks through the size-split thresholds in order.

The walk works like this: for each adjacent pair of denominations in the
sorted list, look up the split threshold using the key pattern
`{COLOR_GROUP}_{lower_index}_{upper_index}`. If the coin's relative size
is below that threshold, the coin belongs to the lower denomination. If
it is above all thresholds, the coin belongs to the largest denomination
in the group.

After picking the denomination, apply the same colour-evidence confidence
boosts as before: high saturation boosts copper confidence, high LAB b*
boosts gold confidence, high hue histogram entropy boosts bicolour
confidence, low saturation boosts silver confidence.

Add two special-case notes: for USD denominations, add a note to the
reasoning list when the dime is selected because its small size is
counterintuitive. For EGP denominations where `50p` or `1La` is selected,
cap the confidence at 0.62 and add a note that these two coins share the
same diameter and cannot be reliably separated by size or colour alone.

**Changes to `classify_coin`:**
Add a call to `_detect_currency` as Stage 0 before the existing colour
group dispatch. Multiply the final confidence by the currency confidence
factor returned by `_detect_currency`.

Change the monetary value lookup from accessing the local `EURO_COINS`
dictionary to calling `get_currency_coins(currency).get(denomination)`.

Add a `currency` field to the returned result dictionary.
Add a `value` field alongside the existing `value_eur` field (keep
`value_eur` with the same value so any downstream code that already
reads `value_eur` continues to work without change).
Add a `symbol` field populated from the coin's registry entry.

**Changes to `calibrate_confidence`:**
Change the reference to `EURO_COINS[den]["diameter_mm"]` to use
`get_currency_coins(currency)[den]["diameter_mm"]`.
Change the reference to the 2e diameter used as a normalising constant
to use `largest_diameter(currency)` instead so it works for any currency.

**Changes to `count_and_total`:**
Instead of accumulating a single total, accumulate per currency.
The returned summary dictionary should contain a key called
`totals_by_currency` where each entry holds the count, total value,
currency symbol, and denomination breakdown for one currency.
Also keep a `grand_total` key that sums everything across currencies
for convenience and testing.
Keep the `total_eur` key with the same value as `grand_total` for
backward compatibility with anything that already reads `total_eur`.

---

## Step 3 — Modify `src/phase6_output.py`

The changes here are smaller than Phase 5. The visual output is unchanged.
Only the console table and some display symbol lookups need updating.

### What to remove

Remove the `LABEL_MAP` dictionary at the top of the file. It only covers
Euro denominations and will be replaced by reading symbols from the registry.

Remove the `SYMBOL_MAP` dictionary at the top of the file. Same reason.

### What to change in `print_table`

The Euro-only version prints one flat table with denomination, count,
unit value, and subtotal, ending with a single TOTAL row in euros.

The new version checks whether the summary contains more than one
currency. If it does, print one section per currency with that currency's
symbol, its denominations, and its own subtotal. If only one currency is
present, the output looks the same as before with a single flat section.

The denomination label in each row should come from the summary data
rather than from a local dictionary, since the summary now carries all
the information needed for display.

The method should still print a grand total of coin count and average
confidence at the bottom regardless of how many currencies are present.

### What to change in `export_json`

Add a `currency` field to each coin entry in the JSON output. In the
Euro-only version each coin only had `value_eur`. In the new version
each coin also has `currency` and `value`. Keep `value_eur` in the
output so any existing scripts that parse the JSON continue to work.

### What to change in `generate_html_report`

Add a `Currency` column to the per-coin results table in the HTML.
The denomination symbols in the table should read from the coin's
`symbol` field in the classified result rather than from a local map.

### What does NOT change in Phase 6

The circle drawing logic, confidence bar drawing, the annotated image
banner, the `save_annotated_image` method, and the overall `run` method
structure are all identical. The `COLORS` dictionary at the top does not
change — the four colour group labels and their BGR values are the same
for all currencies.

---

## Step 4 — Modify `config.yaml`

### What to remove

Remove the `size_splits` block under `classification`. It currently
looks like this:

```
classification:
  size_splits:
    copper_1c_2c:   0.022
    copper_2c_5c:   0.032
    gold_10c_20c:   0.030
    gold_20c_50c:   0.038
    bicolor_1e_2e:  0.043
  low_confidence_threshold: 0.50
```

### What to add

Replace the `classification` section with this:

```
classification:
  currencies: ["EUR"]
  low_confidence_threshold: 0.50
```

That is all that is required for the Euro-only use case. The size splits
are now built into the classifier's default values and do not need to
appear in the config file unless you want to override them.

When you want to switch currency, change the one line:
```
  currencies: ["USD"]
```
or
```
  currencies: ["EGP"]
```
or to use all three:
```
  currencies: ["EUR", "USD", "EGP"]
```

If you need to fine-tune size splits for your specific camera distance,
add them as nested overrides:
```
classification:
  currencies: ["USD"]
  size_splits:
    USD:
      SILVER_0_1: 0.023
      SILVER_1_2: 0.029
      SILVER_2_3: 0.034
  low_confidence_threshold: 0.50
```

The features section of `config.yaml` does not change. The preprocessing,
detection, segmentation, and output sections do not change.

---

## Step 5 — Modify `main.py`

### What to change in `DEFAULT_CONFIG`

Inside the `classification` key of `DEFAULT_CONFIG`, remove the
`size_splits` dictionary with the Euro denomination names. Replace it
with a `currencies` list:

Old block:
```python
"classification": {
    "size_splits": {
        "copper_1c_2c":  0.022,
        "copper_2c_5c":  0.032,
        "gold_10c_20c":  0.030,
        "gold_20c_50c":  0.038,
        "bicolor_1e_2e": 0.043,
    },
    "low_confidence_threshold": 0.50,
},
```

New block:
```python
"classification": {
    "currencies": ["EUR"],
    "low_confidence_threshold": 0.50,
},
```

This keeps the default behaviour identical to the Euro-only version.
Anyone running `python main.py --image coins.jpg` without any config file
will get Euro classification exactly as before.

### What does NOT change in main.py

The argument parser, the `load_config` function, the `_deep_merge` helper,
the `setup_output_dirs` function, the `process_image` function, the
`batch_process` function, and the `parse_args` function are all unchanged.
The pipeline instantiation inside `process_image` is unchanged — it still
calls Phase 1 through Phase 6 in the same order with the same interface.

---

## Step 6 — Verify the migration

After making all the changes above, run these checks in order.

**Check 1: imports resolve**
Run `python -c "from src.coin_registry import COIN_DB; print(COIN_DB.keys())"`.
You should see `dict_keys(['EGP', 'EUR', 'USD'])`.

**Check 2: Euro-only still works**
Run `python main.py --image data/sample_images/test_mixed_coins.jpg`.
The output should be identical to what you got before the migration.
Same denominations, same total, same confidence scores.

**Check 3: USD mode works**
Change `currencies` to `["USD"]` in `config.yaml`, run on an image of
US coins, and check that the output table shows dollar denominations
with the `$` symbol.

**Check 4: EGP mode works**
Change `currencies` to `["EGP"]` in `config.yaml`, run on an image of
Egyptian coins, and check that the output table shows LE denominations.

**Check 5: full test suite**
Run `python -m pytest tests/ -q`. All 90 existing tests should still pass.

---

## New Project Structure

This shows the complete file tree after the migration. Files marked
`[NEW]` were added. Files marked `[CHANGED]` were modified. Files
marked `[SAME]` are identical to the Euro-only version.

```
coin_counter/
│
├── main.py                              [CHANGED]
│   └─ DEFAULT_CONFIG now has currencies instead of Euro split names
│
├── config.yaml                          [CHANGED]
│   └─ classification section now has currencies key instead of split names
│
├── requirements.txt                     [SAME]
├── PRD.md                               [SAME]
├── PROJECT_STRUCTURE.md                 [SAME]
│
├── src/
│   ├── __init__.py                      [SAME]
│   │
│   ├── coin_registry.py                 [NEW]
│   │   └─ COIN_DB dictionary with EUR, USD, EGP coin specs
│   │   └─ Six helper functions: get_currency_coins, get_all_active_coins,
│   │      get_color_groups_for_currency, supported_currencies,
│   │      largest_diameter, currency_symbol
│   │
│   ├── phase1_preprocessing.py          [SAME]
│   │   └─ Image loading, scale normalisation, CLAHE, blur, Canny edges
│   │
│   ├── phase2_detection.py              [SAME]
│   │   └─ Circular Hough Transform, NMS, boundary filtering
│   │
│   ├── phase3_segmentation.py           [SAME]
│   │   └─ ROI crop, circular mask, 128×128 normalisation
│   │
│   ├── phase4_features.py               [SAME]
│   │   └─ Size features, HSV stats, LAB stats, hue histogram,
│   │      colour group detection — works for all currencies unchanged
│   │
│   ├── phase5_classification.py         [CHANGED]
│   │   └─ Removed: EURO_COINS dict, COPPER/GOLD/BICOLOR_BY_SIZE lists,
│   │      _classify_copper, _classify_gold, _classify_bicolor methods,
│   │      Euro-specific size_splits keys
│   │   └─ Added: import from coin_registry, currencies attribute,
│   │      _populate_default_splits method, _detect_currency method (Stage 0),
│   │      _classify_within_group generic method,
│   │      per-currency totals in count_and_total,
│   │      currency and value fields in classify_coin output
│   │
│   ├── phase6_output.py                 [CHANGED]
│   │   └─ Removed: LABEL_MAP dict, SYMBOL_MAP dict (Euro-only)
│   │   └─ Changed: print_table shows per-currency sections,
│   │      JSON export adds currency field per coin,
│   │      HTML report adds Currency column in table
│   │   └─ Unchanged: circle drawing, confidence bars, banner,
│   │      image saving, COLORS dict
│   │
│   └── utils/
│       ├── __init__.py                  [SAME]
│       ├── image_io.py                  [SAME]
│       └── logger.py                    [SAME]
│
├── tests/
│   ├── __init__.py                      [SAME]
│   ├── test_phase3_segmentation.py      [SAME]
│   ├── test_phase4_features.py          [SAME]
│   ├── test_phase5_classification.py    [CHANGED]
│   │   └─ Import EURO_COINS from coin_registry instead of phase5
│   │   └─ Updated rel_size values in a few tests to match new split
│   │      key naming (generic index-based rather than denomination-named)
│   │   └─ Updated total assertions to use grand_total instead of total_eur
│   ├── test_phase6_output.py            [CHANGED]
│   │   └─ sample_summary fixture updated to include totals_by_currency
│   │   └─ print_table assertion updated to match new per-currency format
│   └── test_integration.py              [SAME]
│
├── docs/
│   ├── phase1_details.md                [SAME]
│   ├── phase2_details.md                [SAME]
│   ├── phase3_details.md                [SAME]
│   ├── phase4_details.md                [SAME]
│   ├── phase5_details.md                [SAME]
│   └── phase6_details.md                [SAME]
│
├── data/
│   ├── sample_images/                   [SAME]
│   └── ground_truth/                    [SAME]
│
└── outputs/                             [SAME structure, runtime-generated]
    ├── annotated/
    ├── intermediate/
    └── reports/
```

---

## Summary Table

| File | Action | Reason |
|---|---|---|
| `src/coin_registry.py` | Create from scratch | Central data store for all coin specs across all currencies |
| `src/phase5_classification.py` | Modify | Add Stage 0 currency detection, replace three Euro-specific methods with one generic method, add per-currency totalling |
| `src/phase6_output.py` | Modify | Update console table for multi-currency sections, add currency field to JSON and HTML outputs |
| `config.yaml` | Modify | Replace Euro split key names with a `currencies` list |
| `main.py` | Modify | Replace Euro split key names in DEFAULT_CONFIG with a `currencies` list |
| `src/phase1_preprocessing.py` | Do not change | Pure image processing |
| `src/phase2_detection.py` | Do not change | Pure circle detection |
| `src/phase3_segmentation.py` | Do not change | Pure coin cropping |
| `src/phase4_features.py` | Do not change | Colour and size features work for all currencies |
| `src/utils/image_io.py` | Do not change | File utilities |
| `src/utils/logger.py` | Do not change | Logging setup |
