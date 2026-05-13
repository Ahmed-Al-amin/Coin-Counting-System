# Phase 6: Output & Visualization
## Generic Coin Counting System — Classical CV Pipeline

---

## 1. Overview

The final phase transforms detection and categorization results into human-readable outputs: an annotated image, a console summary table, a JSON results file, and an HTML report. The focus is on coin counts and size categories.

**Goal**: Take all phase results → produce annotated image + JSON + console table + HTML report.

---

## 2. Implementation Steps

### Step 6.1 — Annotated Image Generation

Annotated images now show the size category (Small, Medium, Large) above each validated coin and a confidence bar below. Noise/Non-coin detections are skipped to keep the visualization clean.

### Step 6.2 — Summary Banner

The banner at the top shows:
- **Total Coins Detected**
- Breakdown by size category
- Processing timestamp and average confidence

### Step 6.3 — Data Export

- **JSON**: Contains detailed features, reasoning, and coordinates for every detection.
- **HTML**: A portable report with the embedded annotated image and a detailed table of all detections.
- **Console Table**: A quick summary printed to the terminal showing counts and percentages.

---

## 3. Expected Outputs

| File | Description |
|---|---|
| `outputs/annotated/*.png` | Annotated image with size labels |
| `outputs/reports/*.json` | Machine-readable JSON results |
| `outputs/reports/*.html` | Human-readable HTML report |
| Console | Formatted table printed to stdout |

---

## 4. `config.yaml` Parameters for Phase 6

```yaml
output:
  output_dir: outputs
  generate_html: true
  annotate_confidence_bar: true
  banner_height_px: 60
```
