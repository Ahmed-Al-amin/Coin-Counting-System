# 🪙 Multi-Currency Customisation Guide
### Coin Counting System — EUR · USD · EGP

> **How to make the system work for Euro, US Dollar, Egyptian Pound —
> or any single one of them — and how to add a completely new currency.**

---

## Table of Contents

1. [How the System Was Designed to Change](#1-how-the-system-was-designed-to-change)
2. [Quick-Start: Switching Currency in 30 Seconds](#2-quick-start-switching-currency-in-30-seconds)
3. [Complete Coin Reference Tables](#3-complete-coin-reference-tables)
4. [File-by-File Change Map](#4-file-by-file-change-map)
5. [File 1 — `config.yaml` (always edit this)](#5-file-1--configyaml-always-edit-this)
6. [File 2 — `src/coin_registry.py` (add new currencies here)](#6-file-2--srccoin_registrypy-add-new-currencies-here)
7. [File 3 — `src/phase4_features.py` (colour thresholds)](#7-file-3--srcphase4_featurespy-colour-thresholds)
8. [File 4 — `src/phase5_classification.py` (size splits)](#8-file-4--srcphase5_classificationpy-size-splits)
9. [File 5 — `src/phase6_output.py` (display symbols)](#9-file-5--srcphase6_outputpy-display-symbols)
10. [File 6 — `main.py` (CLI default)](#10-file-6--mainpy-cli-default)
11. [Scenario Walkthroughs](#11-scenario-walkthroughs)
12. [Known Ambiguities & How to Resolve Them](#12-known-ambiguities--how-to-resolve-them)
13. [Adding a Brand-New Currency](#13-adding-a-brand-new-currency)
14. [Testing Your Changes](#14-testing-your-changes)

---

## 1. How the System Was Designed to Change

The pipeline uses a **three-layer separation of concerns**:

```
config.yaml          ← which currencies are active (you change this)
src/coin_registry.py ← physical coin data: diameter, colour, value
src/phase4_features.py ← colour detection thresholds (HSV / LAB)
src/phase5_classification.py ← size-split thresholds per currency
```

**Phases 1, 2, 3 never change** — they deal with image processing
(preprocessing, circle detection, segmentation) which is fully
currency-agnostic.

**Phase 6 never changes** — it only renders whatever Phase 5 returns.

So for most use-cases you only edit **`config.yaml`**.
For a new currency you also edit **`coin_registry.py`** and possibly
fine-tune thresholds in Phase 4 and Phase 5.

### Classification flow (three stages)

```
Feature vector
  │
  ▼  Stage 0: Currency detection
  │   Score each active currency by (colour_group, rel_size) match
  │   → selects "EUR" or "USD" or "EGP"
  │
  ▼  Stage 1: Colour-group dispatch
  │   COPPER → copper group   GOLD → gold group
  │   SILVER → silver group   BICOLOR → bicolour group
  │
  ▼  Stage 2: Size within group
      Walk size-split thresholds for (currency, group)
      → final denomination  e.g. "25c" or "50p"
```

---

## 2. Quick-Start: Switching Currency in 30 Seconds

Open **`config.yaml`** and change exactly **one line**:

```yaml
# Euro only (default)
classification:
  currencies: ["EUR"]

# US Dollar only
classification:
  currencies: ["USD"]

# Egyptian Pound only
classification:
  currencies: ["EGP"]

# All three at once (mixed bowl of coins)
classification:
  currencies: ["EUR", "USD", "EGP"]
```

Then run normally:

```bash
python main.py --image my_coins.jpg
```

That is the **only required change** for supported currencies.
Everything else (size splits, colour thresholds, output symbols) has
correct defaults already loaded from `coin_registry.py`.

---

## 3. Complete Coin Reference Tables

### 3.1 Euro (EUR)

| Denomination | Diameter (mm) | Colour Group | Value | Symbol |
|---|---|---|---|---|
| 1c  | 16.25 | COPPER  | €0.01 | 1¢  |
| 2c  | 18.75 | COPPER  | €0.02 | 2¢  |
| 5c  | 21.25 | COPPER  | €0.05 | 5¢  |
| 10c | 19.75 | GOLD    | €0.10 | 10¢ |
| 20c | 22.25 | GOLD    | €0.20 | 20¢ |
| 50c | 24.25 | GOLD    | €0.50 | 50¢ |
| 1e  | 23.25 | BICOLOR | €1.00 | €1  |
| 2e  | 25.75 | BICOLOR | €2.00 | €2  |

**Within-group size ordering:**
- COPPER: `1c` < `2c` < `5c`
- GOLD: `10c` < `20c` < `50c`
- BICOLOR: `1e` < `2e`

---

### 3.2 US Dollar (USD)

| Denomination | Diameter (mm) | Colour Group | Value | Symbol | Note |
|---|---|---|---|---|---|
| 1c  | 19.05 | COPPER  | $0.01 | 1¢  | Penny — copper-plated zinc |
| 5c  | 21.21 | SILVER  | $0.05 | 5¢  | Nickel — cupronickel |
| 10c | 17.91 | SILVER  | $0.10 | 10¢ | **Dime — SMALLEST by diameter** |
| 25c | 24.26 | SILVER  | $0.25 | 25¢ | Quarter |
| 50c | 30.61 | SILVER  | $0.50 | 50¢ | Half Dollar — largest common coin |
| 1d  | 26.50 | BICOLOR | $1.00 | $1  | Dollar — gold-coloured edge |

> ⚠️ **Critical USD quirk**: the **dime (10¢) is physically smaller than the
> nickel (5¢)**. The size-sort order within SILVER is:
> `10c` (17.91mm) < `5c` (21.21mm) < `25c` (24.26mm) < `50c` (30.61mm).
> The classifier handles this automatically via the `SILVER_0_1` split.

---

### 3.3 Egyptian Pound (EGP)

| Denomination | Diameter (mm) | Colour Group | Value | Symbol | Note |
|---|---|---|---|---|---|
| 25p  | 18.00 | SILVER  | LE 0.25 | 25p  | Older series |
| 50p  | 20.00 | SILVER  | LE 0.50 | 50p  | Older series |
| 1La  | 20.00 | SILVER  | LE 1.00 | LE1* | Older series — same diameter as 50p |
| 50pa | 21.00 | BICOLOR | LE 0.50 | 50p* | Newer 2019 series |
| 1L   | 23.00 | BICOLOR | LE 1.00 | LE1  | Newer 2019 series |

> ⚠️ **EGP ambiguity**: `50p` and `1La` share the same 20.00 mm diameter
> and both are SILVER. They **cannot be separated by size or colour alone**
> without edge-reeding texture analysis (not currently implemented).
> The classifier assigns `50p` by default and flags low confidence (≤ 0.62).

---

### 3.4 Cross-Currency Diameter Conflicts

These coin pairs share similar diameters (≤ 1.5 mm difference) across
currencies. **Colour group is the primary disambiguation signal.**

| Coin A | Coin B | Diameter diff | Resolved by |
|---|---|---|---|
| EUR 50c (24.25mm, GOLD)   | USD 25c (24.26mm, SILVER)  | 0.01 mm | **Colour** |
| EUR 1e  (23.25mm, BICOLOR)| USD 25c (24.26mm, SILVER)  | 1.01 mm | Colour + Size |
| EUR 5c  (21.25mm, COPPER) | USD 5c  (21.21mm, SILVER)  | 0.04 mm | **Colour** |
| EUR 2c  (18.75mm, COPPER) | USD 10c (17.91mm, SILVER)  | 0.84 mm | Colour |
| EUR 1e  (23.25mm, BICOLOR)| EGP 1L  (23.00mm, BICOLOR) | 0.25 mm | Currency Stage 0 |
| USD 25c (24.26mm, SILVER) | EGP 1L  (23.00mm, BICOLOR) | 1.26 mm | Colour |

---

## 4. File-by-File Change Map

This table answers: **"For each scenario, which files do I need to touch?"**

| Scenario | `config.yaml` | `coin_registry.py` | `phase4_features.py` | `phase5_classification.py` | `phase6_output.py` | `main.py` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Switch EUR → USD               | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Switch EUR → EGP               | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Enable all 3 currencies        | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fine-tune size thresholds      | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Fine-tune colour thresholds    | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Add a new currency             | ✅ | ✅ | maybe | ✅ | ❌ | ❌ |
| Change CLI default currency    | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Change output symbol (e.g. LE) | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## 5. File 1 — `config.yaml` (always edit this)

`config.yaml` is the **only file you need for day-to-day currency switching**.

### 5.1 The `currencies` key

```yaml
classification:
  currencies: ["EUR"]           # Euro only
  currencies: ["USD"]           # US Dollar only
  currencies: ["EGP"]           # Egyptian Pound only
  currencies: ["EUR", "USD"]    # Mixed EUR + USD
  currencies: ["EUR", "USD", "EGP"]  # All three
```

### 5.2 Size-split overrides (optional fine-tuning)

The defaults work for images taken 30–50 cm above a flat surface with a
smartphone. If your setup produces different pixel sizes, override per currency:

```yaml
classification:
  currencies: ["USD"]
  size_splits:
    USD:
      SILVER_0_1: 0.023    # 10c / 5c boundary  (default: 0.025)
      SILVER_1_2: 0.029    # 5c  / 25c boundary (default: 0.030)
      SILVER_2_3: 0.034    # 25c / 50c boundary (default: 0.036)
```

```yaml
classification:
  currencies: ["EGP"]
  size_splits:
    EGP:
      SILVER_0_1:  0.024   # 25p / 50p boundary  (default: 0.025)
      BICOLOR_0_1: 0.027   # 50pa / 1L boundary  (default: 0.028)
```

```yaml
classification:
  currencies: ["EUR"]
  size_splits:
    EUR:
      COPPER_0_1:  0.021   # 1c / 2c  (default: 0.022)
      COPPER_1_2:  0.027   # 2c / 5c  (default: 0.028)
      GOLD_0_1:    0.026   # 10c / 20c (default: 0.027)
      GOLD_1_2:    0.029   # 20c / 50c (default: 0.030)
      BICOLOR_0_1: 0.037   # 1e / 2e  (default: 0.038)
```

### 5.3 How split keys are named

```
{COLOR_GROUP}_{lower_index}_{upper_index}

Where denominations within each group are sorted ascending by diameter:

EUR COPPER:   index 0=1c  index 1=2c  index 2=5c
              COPPER_0_1 separates 1c from 2c
              COPPER_1_2 separates 2c from 5c

EUR GOLD:     index 0=10c  index 1=20c  index 2=50c
              GOLD_0_1 separates 10c from 20c

USD SILVER:   index 0=10c  index 1=5c  index 2=25c  index 3=50c
              (sorted by diameter: 17.91 < 21.21 < 24.26 < 30.61)
              SILVER_0_1 separates 10c from 5c

EGP BICOLOR:  index 0=50pa  index 1=1L
              BICOLOR_0_1 separates 50pa from 1L
```

### 5.4 How to calibrate split values for your camera

```bash
# 1. Run with --debug on a known image where you can identify each coin
python main.py --image known_coins.jpg --debug --output-dir debug_out

# 2. Look at outputs/intermediate/phase4/coin_XXX_features.png
#    Each panel shows rel_size for that coin

# 3. Set split = midpoint between the two rel_size values you observe
#    Example: 10c has rel_size=0.021, 5c has rel_size=0.028
#    Set SILVER_0_1 = (0.021 + 0.028) / 2 = 0.024
```

---

## 6. File 2 — `src/coin_registry.py` (add new currencies here)

This is the **single source of truth** for all coin physical properties.
You only touch this file when adding a new currency.

### 6.1 Structure of `COIN_DB`

```python
COIN_DB = {
    "CURRENCY_CODE": {
        "denomination_key": {
            "value":           float,   # monetary value in base unit
            "diameter_mm":     float,   # physical outer diameter
            "color_group":     str,     # COPPER | GOLD | SILVER | BICOLOR
            "symbol":          str,     # displayed in UI / HTML
            "currency_symbol": str,     # €, $, LE, ...
            "notes":           str,     # optional disambiguation note
        },
        ...
    },
    ...
}
```

### 6.2 Adding a new currency — example (British Pound GBP)

```python
# In src/coin_registry.py, inside COIN_DB = { ... }

"GBP": {
    "1p":   {"value": 0.01, "diameter_mm": 20.30, "color_group": "COPPER",
             "symbol": "1p",  "currency_symbol": "£"},
    "2p":   {"value": 0.02, "diameter_mm": 25.90, "color_group": "COPPER",
             "symbol": "2p",  "currency_symbol": "£"},
    "5p":   {"value": 0.05, "diameter_mm": 18.00, "color_group": "SILVER",
             "symbol": "5p",  "currency_symbol": "£"},
    "10p":  {"value": 0.10, "diameter_mm": 24.50, "color_group": "SILVER",
             "symbol": "10p", "currency_symbol": "£"},
    "20p":  {"value": 0.20, "diameter_mm": 21.40, "color_group": "SILVER",
             "symbol": "20p", "currency_symbol": "£"},
    "50p":  {"value": 0.50, "diameter_mm": 27.30, "color_group": "SILVER",
             "symbol": "50p", "currency_symbol": "£"},
    "1L":   {"value": 1.00, "diameter_mm": 23.43, "color_group": "BICOLOR",
             "symbol": "£1",  "currency_symbol": "£"},
    "2L":   {"value": 2.00, "diameter_mm": 28.40, "color_group": "BICOLOR",
             "symbol": "£2",  "currency_symbol": "£"},
},
```

**Rules for colour_group assignment:**

| Metal / Alloy | Assign |
|---|---|
| Copper-plated steel or zinc | `COPPER` |
| Nordic Gold / brass / gold-toned | `GOLD` |
| Cupronickel / nickel-plated steel / stainless steel (silver-grey) | `SILVER` |
| Two-tone: one ring + different centre | `BICOLOR` |

---

## 7. File 3 — `src/phase4_features.py` (colour thresholds)

**When to change:** when coins of your target currency look visually
different from standard Euro coins — for example if your coins are
very worn, photographed under unusual lighting, or if you are adding a
currency with a colour not covered by the current four groups.

### 7.1 Where to look

Open `src/phase4_features.py` and find the method `classify_color_group`.
The decision tree starts at approximately line 310.

### 7.2 Current thresholds and what they mean

```python
# ── BICOLOR check (must come first) ──────────────────────────────────────
# hue_std > 22  means the coin contains pixels spanning many hue angles
# → two colour zones (gold rim + silver centre, or vice-versa)
if hue_std > bicolor_hue_std_thresh:          # default: 22
    return "BICOLOR"

# ── COPPER ───────────────────────────────────────────────────────────────
# Hue 8–26°   orange-brown range in OpenCV HSV (0–180 scale)
# sat > 65    vivid — copper has strong colour vs. silver which is grey
# LAB a* > 132  red-shifted: copper alloy shifts toward red in CIELAB
if (8.0 <= hue_mean <= 26.0 and sat_mean > 65.0 and lab_a > 132.0):
    return "COPPER"

# ── GOLD ─────────────────────────────────────────────────────────────────
# Hue 14–42°  yellow-gold range
# sat > 28    some gold toning (Nordic Gold is less vivid than copper)
# LAB b* > 133  yellow-shifted: gold alloy shifts toward yellow in CIELAB
if (14.0 <= hue_mean <= 42.0 and sat_mean > 28.0 and lab_b > 133.0):
    return "GOLD"

# ── SILVER ───────────────────────────────────────────────────────────────
# sat < 50    near-grey: cupronickel has very low colour saturation
if sat_mean < 50.0:
    return "SILVER"
```

### 7.3 What to adjust per currency

**For USD** — no changes needed. USD penny is COPPER (identical to EUR copper
cents), dime/nickel/quarter/half are SILVER (cupronickel), dollar is BICOLOR.
The four groups map perfectly.

**For EGP** — no changes needed. Egyptian coins are either silver-grey
(SILVER) or two-tone (BICOLOR). The existing SILVER and BICOLOR detection
works as-is.

**When to lower the BICOLOR hue_std threshold:**

```yaml
# In config.yaml — lower = more coins classified as BICOLOR
features:
  bicolor_hue_std_thresh: 18    # more sensitive (default: 22)
```

Use this if your camera angle emphasises the ring/centre contrast.
Lower it to 15–18 for top-down macro shots where the two zones are sharp.

**When to widen the COPPER hue range:**

```python
# In src/phase4_features.py, classify_color_group():
# Change:
if (8.0 <= hue_mean <= 26.0 ...):
# To (for very worn/dark copper coins):
if (6.0 <= hue_mean <= 30.0 ...):
```

---

## 8. File 4 — `src/phase5_classification.py` (size splits)

**When to change:** you should almost never edit this file directly.
Use `config.yaml → classification → size_splits` instead (Section 5.2).

The only reason to edit this file is if you want to **change the default
fallback values** that apply when no config override exists — for example
if you're packaging the system for a specific camera rig.

### 8.1 Where the defaults live

```python
# src/phase5_classification.py, method _populate_default_splits()

defaults = {
    "EUR": {
        "COPPER_0_1":  0.022,    # 1c  | 2c  boundary
        "COPPER_1_2":  0.028,    # 2c  | 5c  boundary
        "GOLD_0_1":    0.027,    # 10c | 20c boundary
        "GOLD_1_2":    0.030,    # 20c | 50c boundary
        "BICOLOR_0_1": 0.038,    # 1e  | 2e  boundary
    },
    "USD": {
        "SILVER_0_1":  0.025,    # 10c | 5c  boundary (dime is smaller!)
        "SILVER_1_2":  0.030,    # 5c  | 25c boundary
        "SILVER_2_3":  0.036,    # 25c | 50c boundary
    },
    "EGP": {
        "SILVER_0_1":  0.025,    # 25p | 50p boundary
        "BICOLOR_0_1": 0.028,    # 50pa | 1L  boundary
    },
}
```

### 8.2 How split values are derived

```
rel_size = coin_radius_px / image_diagonal_px

For a reference image where image_diagonal ≈ 1448 px (1024×1024):
    20 mm coin → radius ≈ 53 px → rel_size ≈ 0.037

Split between denomination A (diam=d1) and B (diam=d2):
    midpoint_mm  = (d1 + d2) / 2
    split_rel    = midpoint_mm / largest_coin_mm * max_rel_size_of_largest

Simpler rule of thumb:
    split_rel ≈ midpoint_mm × 0.00135
    (where 0.00135 = calibration constant for 30cm camera height)
```

---

## 9. File 5 — `src/phase6_output.py` (display symbols)

**When to change:** the output symbols (what appears on the annotated image
and in the HTML report) come directly from `coin_registry.py`'s `symbol`
field. You only edit `phase6_output.py` if you want to change:

- Annotation circle colours per colour group
- The confidence bar colour thresholds
- Whether the denomination or the monetary value appears as the label

### 9.1 Annotation circle colours

```python
# src/phase6_output.py, top of file
COLORS: Dict[str, tuple] = {
    "COPPER":  (0,   110, 210),   # BGR — vivid orange
    "GOLD":    (0,   210, 215),   # BGR — yellow-gold
    "BICOLOR": (200, 80,  20),    # BGR — blue-purple
    "SILVER":  (200, 200, 200),   # BGR — light grey
    "UNKNOWN": (0,   0,   230),   # BGR — red
}
```

Change any BGR tuple to adjust annotation circle colour.
These four groups cover all three currencies — no change needed
when switching currencies.

### 9.2 Label symbols

The annotation label on each coin comes from `coin_registry.py`:

```python
# In coin_registry.py, each denomination entry:
"symbol": "25¢"    # ← this appears on the annotated image

# To change what appears on the image for USD quarter:
"25c": {"value": 0.25, ..., "symbol": "Quarter"},  # custom label
```

---

## 10. File 6 — `main.py` (CLI default)

**When to change:** if you want to bake in a default currency so users
don't have to specify it every time.

### 10.1 The relevant section

```python
# main.py, DEFAULT_CONFIG dict (~line 30)

DEFAULT_CONFIG = {
    ...
    "classification": {
        "currencies": ["EUR"],      # ← change this line
        "size_splits": { ... },
        ...
    },
    ...
}
```

Change `["EUR"]` to `["USD"]`, `["EGP"]`, or `["EUR", "USD", "EGP"]`.

This only affects runs where no `--config` flag is passed. If a
`config.yaml` is provided on the CLI, its value takes priority.

---

## 11. Scenario Walkthroughs

### Scenario A — Egyptian Pound only

**Use case:** ATM cash counting in Egypt, only EGP coins in frame.

**Step 1** — Edit `config.yaml`:
```yaml
classification:
  currencies: ["EGP"]
```

**Step 2** — Run:
```bash
python main.py --image egp_coins.jpg
```

**Expected output table:**
```
══════════════════════════════════════════════════════
  💰  COIN COUNTING RESULTS
══════════════════════════════════════════════════════
  Denomination    Count     Unit    Subtotal
──────────────────────────────────────────────────────
  25p                 3   LE0.25     LE0.75
  50pa                2   LE0.50     LE1.00
  1L                  2   LE1.00     LE2.00
──────────────────────────────────────────────────────
  Subtotal            7             LE3.75
══════════════════════════════════════════════════════
  Total coins:        7
  Average confidence: 71%
```

**Limitations to be aware of:**
- `50p` (older series) and `1La` (older 1-pound) share 20 mm — confidence ≤ 0.62 for either
- The system labels them `50p` by default; inspect the low-confidence flag in the report
- Both 2019-series bicolour coins (`50pa`, `1L`) work reliably

---

### Scenario B — US Dollar only

**Use case:** Vending machine coin tray in the United States.

**Step 1** — Edit `config.yaml`:
```yaml
classification:
  currencies: ["USD"]
```

**Step 2** — Run:
```bash
python main.py --image usd_coins.jpg
```

**Important:** the dime (10¢) is physically the **smallest** USD coin.
If your image has dimes and nickels close together, the classifier uses
the `SILVER_0_1` split (default 0.025) to separate them. If they look
the same size in your photo (similar viewing distance), tune:

```yaml
classification:
  currencies: ["USD"]
  size_splits:
    USD:
      SILVER_0_1: 0.023    # lower if dimes look too large in your photos
```

---

### Scenario C — Euro only (default, unchanged)

No changes required. The default `config.yaml` already has:

```yaml
classification:
  currencies: ["EUR"]
```

---

### Scenario D — Mixed bowl with EUR + USD + EGP

**Use case:** currency exchange desk, tourist pocket change, research lab.

**Step 1** — Edit `config.yaml`:
```yaml
classification:
  currencies: ["EUR", "USD", "EGP"]
```

**Step 2** — Run:
```bash
python main.py --image mixed_coins.jpg
```

**How disambiguation works:**
- GOLD coins → always EUR (only EUR uses Nordic Gold)
- COPPER coins → EUR 1c/2c/5c vs. USD 1c — separated by colour subtlety (EUR copper is warmer orange, USD penny is brighter copper) and size
- SILVER coins → USD (dime, nickel, quarter, half) vs. EGP (25p, 50p, 1La) — separated by size
- BICOLOR coins → EUR (€1, €2) vs. EGP (50pa, 1L) vs. USD ($1) — separated by size and hue pattern

**Output table shows per-currency sections:**
```
══════════════════════════════════════════════════════════
  💰  COIN COUNTING RESULTS
══════════════════════════════════════════════════════════
  ── EUR ──
  Denomination    Count     Unit    Subtotal
────────────────────────────────────────────────────────
  50c                 1    €0.50     €0.50
  1e                  1    €1.00     €1.00
────────────────────────────────────────────────────────
  Subtotal            2             €1.50

  ── USD ──
  Denomination    Count     Unit    Subtotal
────────────────────────────────────────────────────────
  10c                 2    $0.10     $0.20
  25c                 1    $0.25     $0.25
────────────────────────────────────────────────────────
  Subtotal            3             $0.45

  ── EGP ──
  Denomination    Count     Unit    Subtotal
────────────────────────────────────────────────────────
  1L                  2   LE1.00    LE2.00
────────────────────────────────────────────────────────
  Subtotal            2             LE2.00
══════════════════════════════════════════════════════════
  Total coins:        7
  Average confidence: 68%
```

> Note: totals are shown **per currency** because combining EUR + USD + EGP
> into one number has no financial meaning.

---

## 12. Known Ambiguities & How to Resolve Them

| Pair | Issue | Current handling | Better fix |
|---|---|---|---|
| EGP 50p vs EGP 1La | Identical diameter (20 mm), both SILVER | Label as 50p, conf ≤ 0.62 | Use edge-reeding or obverse texture analysis |
| EUR 2c vs USD 10c | Similar size (18.75 vs 17.91 mm), different colour (COPPER vs SILVER) | Colour resolves it | Ensure good lighting so saturation is measurable |
| EUR 50c vs USD 25c | Nearly identical diameter (24.25 vs 24.26 mm) | Colour resolves it: GOLD vs SILVER | Ensure white-balanced image |
| EUR 1e vs EGP 1L | Similar diameter (23.25 vs 23.00 mm), both BICOLOR | Currency Stage 0 heuristic | Use currency=["EUR"] or ["EGP"] not both |
| USD 1c vs EUR 10c | Copper vs Gold, close in size (19.05 vs 19.75 mm) | Colour resolves | Check CLAHE is enabled for even lighting |

**General rule:** when only one currency can appear in the image, always set
`currencies: ["EUR"]` (or USD or EGP). Multi-currency mode is for genuinely
mixed bowls and always has lower average confidence.

---

## 13. Adding a Brand-New Currency

Follow these five steps in order:

### Step 1 — Add to `src/coin_registry.py`

```python
COIN_DB["GBP"] = {
    "1p":  {"value": 0.01, "diameter_mm": 20.30, "color_group": "COPPER",
            "symbol": "1p", "currency_symbol": "£"},
    # ... all denominations
}
```

### Step 2 — Enable in `config.yaml`

```yaml
classification:
  currencies: ["GBP"]
```

### Step 3 — Add size-split defaults to `src/phase5_classification.py`

In `_populate_default_splits()`:

```python
"GBP": {
    # COPPER: 1p(20.30) | 2p(25.90) — split at midpoint 23.1mm → rel ≈ 0.031
    "COPPER_0_1":  0.031,
    # SILVER: 5p(18.0) < 20p(21.4) < 10p(24.5) < 50p(27.3)
    "SILVER_0_1":  0.026,   # 5p  | 20p
    "SILVER_1_2":  0.031,   # 20p | 10p
    "SILVER_2_3":  0.036,   # 10p | 50p
    # BICOLOR: 1L(23.43) | 2L(28.40)
    "BICOLOR_0_1": 0.035,
},
```

### Step 4 — (Optional) Tune colour thresholds in `phase4_features.py`

Only needed if the new currency has unusual metal colours not covered by
the current COPPER / GOLD / SILVER / BICOLOR groups.

### Step 5 — Write tests

```python
# tests/test_phase5_classification.py — add:
def test_classify_gbp_penny(classifier_gbp):
    feat   = make_features("COPPER", rel_size=0.028, ...)
    result = classifier_gbp.classify_coin(feat)
    assert result["currency"]     == "GBP"
    assert result["denomination"] == "1p"
```

---

## 14. Testing Your Changes

### Quick sanity check (30 seconds)

```bash
# After changing config.yaml, run the built-in test suite:
python -m pytest tests/ -q

# All 90 tests should still pass — they test the framework,
# not currency-specific thresholds.
```

### Manual calibration run

```bash
# Run with --debug to see per-coin feature values:
python main.py --image your_coins.jpg --debug --output-dir debug/

# Then open: debug/intermediate/phase4/coin_000_features.png
# Check:
#   - color_group matches what you expect visually
#   - rel_size values are spread enough to separate denominations
#   - size_bin makes sense for the coin size
```

### Checking size splits are correct

```python
# Quick Python check — paste in a terminal:
from src.phase5_classification import CoinClassifier
clf = CoinClassifier({"currencies": ["USD"]})
print(clf.size_splits)
# Should show USD SILVER_0_1=0.025 etc.
```

### Regression test for totals

```python
# Verify total is computed correctly for a known set:
from src.phase5_classification import CoinClassifier
clf = CoinClassifier({"currencies": ["USD"]})

fake_coins = [
    {"denomination": "25c", "value": 0.25, "confidence": 0.80},
    {"denomination": "10c", "value": 0.10, "confidence": 0.80},
    {"denomination": "25c", "value": 0.25, "confidence": 0.80},
]
summary = clf.count_and_total(fake_coins)
assert summary["grand_total"] == 0.60, f"Got {summary['grand_total']}"
print("✅ Total correct")
```

---

*Coin Counting System — Multi-Currency Customisation Guide v1.0*
*Covers: EUR (8 coins) · USD (6 coins) · EGP (5 coins)*
