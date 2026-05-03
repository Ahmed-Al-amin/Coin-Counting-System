# Phase 6: Output & Visualization
## Coin Counting System — Classical CV Pipeline

---

## 1. Overview

The final phase transforms classification results into human-readable outputs: an annotated image, a console summary table, a JSON results file, and an optional HTML report. Clear visualization is critical for debugging and for demonstrating the system's accuracy.

**Goal**: Take all phase results → produce annotated image + JSON + console table + optional HTML report.

---

## 2. Implementation Steps

### Step 6.1 — Annotated Image Generation

```python
# src/phase6_output.py
import cv2
import numpy as np
import json
import os
from typing import List, Dict
from datetime import datetime
from pathlib import Path

# Color scheme for annotation overlays (BGR)
COLORS = {
    'COPPER':  (0, 100, 200),    # Orange-red (BGR)
    'GOLD':    (0, 200, 210),    # Yellow-gold (BGR)
    'BICOLOR': (200, 100, 0),    # Blue-purple (BGR)
    'SILVER':  (200, 200, 200),  # Light gray
    'UNKNOWN': (0, 0, 255),      # Red for unknowns
}

LABEL_MAP = {
    '1c':  '1¢',   '2c':  '2¢',   '5c':  '5¢',
    '10c': '10¢',  '20c': '20¢',  '50c': '50¢',
    '1e':  '€1',   '2e':  '€2',   'UNKNOWN': '?',
}

class ResultExporter:
    def __init__(self, config: dict, output_dir: str = "outputs"):
        self.config     = config
        self.output_dir = output_dir
        self.run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")

    def draw_annotations(self, image: np.ndarray, 
                          classified_coins: List[Dict]) -> np.ndarray:
        """
        Draw annotated overlays on image:
        - Colored circle outline (color = coin denomination group)
        - Label with denomination symbol (e.g., "€1") above coin
        - Confidence bar below coin (green=high, red=low)
        - Small center dot
        """
        annotated = image.copy()
        
        for coin in classified_coins:
            x, y, r = coin['circle']
            den      = coin['denomination']
            conf     = coin['confidence']
            cgroup   = coin['color_group']
            
            color = COLORS.get(cgroup, COLORS['UNKNOWN'])
            
            # ── Outer circle ──────────────────────────────────────────
            thickness = max(2, r // 20)
            cv2.circle(annotated, (x, y), r, color, thickness)
            
            # ── Center dot ────────────────────────────────────────────
            cv2.circle(annotated, (x, y), 3, color, -1)
            
            # ── Denomination label ────────────────────────────────────
            label = LABEL_MAP.get(den, '?')
            font       = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.5, r / 60)
            font_thick = max(1, int(r / 30))
            
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
            
            # Label position: centered above the circle
            lx = x - tw // 2
            ly = y - r - 8
            
            # Background rectangle for readability
            pad = 4
            cv2.rectangle(annotated,
                          (lx - pad, ly - th - pad),
                          (lx + tw + pad, ly + baseline + pad),
                          (0, 0, 0), -1)
            cv2.putText(annotated, label, (lx, ly),
                        font, font_scale, color, font_thick, cv2.LINE_AA)
            
            # ── Confidence bar ────────────────────────────────────────
            bar_w = int(r * 1.5)
            bar_h = max(4, r // 10)
            bar_x = x - bar_w // 2
            bar_y = y + r + 6
            
            # Background (dark gray)
            cv2.rectangle(annotated, (bar_x, bar_y), 
                          (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            
            # Filled portion (green → yellow → red by confidence)
            fill_w = int(bar_w * conf)
            bar_color = self._confidence_color(conf)
            cv2.rectangle(annotated, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
            
            # Confidence percentage
            conf_label = f"{int(conf*100)}%"
            cv2.putText(annotated, conf_label, 
                        (bar_x + bar_w + 4, bar_y + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        
        return annotated

    def _confidence_color(self, conf: float) -> tuple:
        """Return BGR color: green (high conf) → yellow → red (low conf)."""
        if conf >= 0.75:
            return (0, 200, 0)     # Green
        elif conf >= 0.50:
            return (0, 200, 200)   # Yellow
        else:
            return (0, 0, 200)     # Red
```

### Step 6.2 — Summary Banner

```python
    def draw_summary_banner(self, annotated: np.ndarray, 
                             summary: Dict) -> np.ndarray:
        """
        Draw a summary banner at the top of the annotated image showing:
        - Total value in EUR
        - Total coin count
        - Processing timestamp
        """
        h, w = annotated.shape[:2]
        banner_h = 60
        
        # Create banner background
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
        banner[:] = (20, 20, 20)  # Dark background
        
        # Total value (large text)
        total_text = f"Total: EUR {summary['total_eur']:.2f}"
        cv2.putText(banner, total_text, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 230, 100), 2, cv2.LINE_AA)
        
        # Coin count
        count_text = f"{summary['total_coins']} coins"
        cv2.putText(banner, count_text, (w - 180, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Avg confidence
        conf_text = f"Avg conf: {summary['avg_confidence']:.0%}"
        cv2.putText(banner, conf_text, (w - 350, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
        
        return np.vstack([banner, annotated])
```

### Step 6.3 — Save Annotated Image

```python
    def save_annotated_image(self, annotated: np.ndarray, 
                              source_path: str) -> str:
        """Save annotated result image to outputs/annotated/."""
        stem = Path(source_path).stem
        out_path = f"{self.output_dir}/annotated/{stem}_{self.run_id}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated)
        return out_path
```

### Step 6.4 — JSON Export

```python
    def export_json(self, phase5_result: dict, 
                    source_path: str) -> str:
        """
        Export full results to JSON for programmatic use.
        
        JSON schema:
        {
          "run_id": "...",
          "source_image": "...",
          "timestamp": "...",
          "summary": { "total_eur": ..., "total_coins": ..., ... },
          "coins": [ { "denomination": ..., "value_eur": ..., ... }, ... ]
        }
        """
        results = {
            "run_id":       self.run_id,
            "source_image": str(source_path),
            "timestamp":    datetime.now().isoformat(),
            "summary":      phase5_result['summary'],
            "coins": [
                {
                    "coin_index":   c['coin_index'],
                    "denomination": c['denomination'],
                    "value_eur":    c['value_eur'],
                    "confidence":   round(c['confidence'], 4),
                    "color_group":  c['color_group'],
                    "circle": {
                        "x": c['circle'][0],
                        "y": c['circle'][1],
                        "r": c['circle'][2],
                    },
                    "reasoning": c['reasoning'],
                }
                for c in phase5_result['classified_coins']
            ]
        }
        
        stem     = Path(source_path).stem
        out_path = f"{self.output_dir}/reports/{stem}_{self.run_id}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return out_path
```

### Step 6.5 — Console Table

```python
    def print_table(self, phase5_result: dict) -> str:
        """
        Print and return a formatted console summary table.
        
        Example output:
        ╔══════════════════════════════════════════════╗
        ║          COIN COUNTING RESULTS               ║
        ╠══════════╦═══════╦══════════╦═══════════════╣
        ║ Coin     ║ Count ║ Unit Val ║ Subtotal      ║
        ╠══════════╬═══════╬══════════╬═══════════════╣
        ║ 1¢       ║   1   ║  €0.01   ║ €0.01         ║
        ║ 20¢      ║   2   ║  €0.20   ║ €0.40         ║
        ║ 50¢      ║   1   ║  €0.50   ║ €0.50         ║
        ║ €1       ║   1   ║  €1.00   ║ €1.00         ║
        ╠══════════╬═══════╬══════════╬═══════════════╣
        ║ TOTAL    ║   5   ║          ║ €1.91         ║
        ╚══════════╩═══════╩══════════╩═══════════════╝
        """
        summary   = phase5_result['summary']
        breakdown = summary['breakdown']
        
        lines = []
        lines.append("\n" + "═" * 54)
        lines.append("  💰  COIN COUNTING RESULTS")
        lines.append("═" * 54)
        lines.append(f"  {'Denomination':<14} {'Count':>6} {'Unit':>8} {'Subtotal':>10}")
        lines.append("─" * 54)
        
        for row in breakdown:
            den_sym  = LABEL_MAP.get(row['denomination'], row['denomination'])
            count    = row['count']
            unit_val = f"€{row['unit_value']:.2f}"
            subtotal = f"€{row['subtotal_eur']:.2f}"
            lines.append(f"  {den_sym:<14} {count:>6} {unit_val:>8} {subtotal:>10}")
        
        lines.append("─" * 54)
        lines.append(f"  {'TOTAL':<14} {summary['total_coins']:>6} {'':>8} "
                     f"€{summary['total_eur']:.2f}".rjust(10))
        lines.append("═" * 54)
        lines.append(f"  Average confidence: {summary['avg_confidence']:.1%}")
        if summary['low_confidence_count'] > 0:
            lines.append(f"  ⚠️  Low-confidence detections: "
                         f"{summary['low_confidence_count']}")
        lines.append("")
        
        output = "\n".join(lines)
        print(output)
        return output
```

### Step 6.6 — HTML Report

```python
    def generate_html_report(self, phase5_result: dict,
                              annotated_path: str,
                              source_path: str) -> str:
        """Generate a self-contained HTML report with embedded image."""
        summary   = phase5_result['summary']
        coins     = phase5_result['classified_coins']
        
        # Embed annotated image as base64
        import base64
        with open(annotated_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        rows = ""
        for coin in coins:
            den  = LABEL_MAP.get(coin['denomination'], '?')
            conf = f"{coin['confidence']:.0%}"
            conf_color = '#2ecc71' if coin['confidence'] >= 0.75 else \
                         '#f39c12' if coin['confidence'] >= 0.50 else '#e74c3c'
            rows += f"""
            <tr>
              <td>{coin['coin_index']}</td>
              <td><strong>{den}</strong></td>
              <td>€{coin['value_eur']:.2f}</td>
              <td style="color:{conf_color}; font-weight:bold">{conf}</td>
              <td>{coin['color_group']}</td>
              <td><small>{'; '.join(coin['reasoning'])}</small></td>
            </tr>"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Coin Counter Report — {self.run_id}</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; 
            margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
    h1   {{ color: #f0c040; text-align: center; letter-spacing: 2px; }}
    .summary {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; }}
    .card {{ background: #16213e; border-radius: 12px; padding: 20px 30px;
             text-align: center; border: 1px solid #0f3460; }}
    .card .value {{ font-size: 2em; color: #f0c040; font-weight: bold; }}
    .card .label {{ color: #aaa; font-size: 0.9em; }}
    img   {{ max-width: 100%; border-radius: 8px; border: 2px solid #0f3460; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th    {{ background: #0f3460; padding: 10px; text-align: left; }}
    td    {{ padding: 8px 10px; border-bottom: 1px solid #333; }}
    tr:hover {{ background: #16213e; }}
  </style>
</head>
<body>
  <h1>🪙 Coin Counting Report</h1>
  <p style="text-align:center;color:#aaa">
    Source: {Path(source_path).name} | Run: {self.run_id}
  </p>

  <div class="summary">
    <div class="card">
      <div class="value">€{summary['total_eur']:.2f}</div>
      <div class="label">Total Value</div>
    </div>
    <div class="card">
      <div class="value">{summary['total_coins']}</div>
      <div class="label">Coins Detected</div>
    </div>
    <div class="card">
      <div class="value">{summary['avg_confidence']:.0%}</div>
      <div class="label">Avg Confidence</div>
    </div>
  </div>

  <img src="data:image/png;base64,{img_b64}" alt="Annotated coin image">

  <h2>Per-Coin Results</h2>
  <table>
    <tr><th>#</th><th>Denomination</th><th>Value</th>
        <th>Confidence</th><th>Color Group</th><th>Reasoning</th></tr>
    {rows}
  </table>
</body>
</html>"""
        
        stem     = Path(source_path).stem
        out_path = f"{self.output_dir}/reports/{stem}_{self.run_id}.html"
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return out_path

    def run(self, all_phase_results: dict, source_path: str) -> dict:
        """Execute all output generation."""
        p1, p2, p3, p4, p5 = (
            all_phase_results['phase1'],
            all_phase_results['phase2'],
            all_phase_results['phase3'],
            all_phase_results['phase4'],
            all_phase_results['phase5'],
        )
        
        # Annotate image
        annotated = self.draw_annotations(p1['scaled'], p5['classified_coins'])
        annotated = self.draw_summary_banner(annotated, p5['summary'])
        
        # Save outputs
        img_path  = self.save_annotated_image(annotated, source_path)
        json_path = self.export_json(p5, source_path)
        self.print_table(p5)
        
        html_path = None
        if self.config.get('generate_html', True):
            html_path = self.generate_html_report(p5, img_path, source_path)
        
        return {
            'annotated_image': annotated,
            'annotated_path':  img_path,
            'json_path':       json_path,
            'html_path':       html_path,
        }
```

---

## 3. Strategy Explanation

### 3.1 Color Coding by Metal Group

Using the same 4 colors throughout (circle outline + bar + HTML) creates a consistent visual grammar — users quickly learn that blue-green = gold coins, red-orange = copper, etc.

### 3.2 Confidence Bar Design

A continuous bar (rather than a simple number) makes confidence immediately scannable across many coins. The color shift (green → yellow → red) maps to traffic light semantics users intuitively understand.

### 3.3 Self-Contained HTML

Embedding the annotated image as base64 makes the HTML report a **single portable file** — no external dependencies, sharable via email or USB without broken image references.

### 3.4 JSON Schema for Integration

The JSON output is designed for downstream integration:
- Automation scripts can parse totals directly
- Ground truth comparison tools can match by `circle.x, circle.y`
- CI/CD pipelines can assert `total_eur` matches expected value

---

## 4. `config.yaml` Parameters for Phase 6

```yaml
output:
  output_dir: outputs
  generate_html: true
  annotate_confidence_bar: true
  annotate_label: true
  banner_height_px: 60
  min_confidence_to_annotate: 0.0   # Set > 0 to skip very uncertain coins
```

---

## 5. Expected Outputs

| File | Description |
|---|---|
| `outputs/annotated/{stem}_{run_id}.png` | Annotated image with labels |
| `outputs/reports/{stem}_{run_id}.json` | Machine-readable JSON results |
| `outputs/reports/{stem}_{run_id}.html` | Human-readable HTML report |
| Console | Formatted table printed to stdout |

---

## 6. Test Cases (Phase 6 Complete When All Pass)

```python
# tests/test_phase6_output.py
import pytest
import cv2
import json
import numpy as np
from pathlib import Path
from src.phase6_output import ResultExporter, LABEL_MAP, COLORS

CONFIG = {'generate_html': True}

@pytest.fixture
def exporter(tmp_path):
    return ResultExporter(config=CONFIG, output_dir=str(tmp_path))

@pytest.fixture
def sample_image():
    return np.zeros((400, 400, 3), dtype=np.uint8)

@pytest.fixture
def sample_classified_coins():
    return [
        {'coin_index': 0, 'circle': (100, 100, 40), 'denomination': '50c',
         'value_eur': 0.50, 'confidence': 0.85, 'color_group': 'GOLD',
         'reasoning': ['Color group: GOLD', 'Large gold → 50c']},
        {'coin_index': 1, 'circle': (250, 200, 30), 'denomination': '1c',
         'value_eur': 0.01, 'confidence': 0.72, 'color_group': 'COPPER',
         'reasoning': ['Color group: COPPER', 'Small → 1c']},
    ]

@pytest.fixture
def sample_summary():
    return {
        'total_eur': 0.51, 'total_coins': 2, 'avg_confidence': 0.785,
        'low_confidence_count': 0,
        'breakdown': [
            {'denomination': '1c',  'count': 1, 'unit_value': 0.01, 'subtotal_eur': 0.01},
            {'denomination': '50c', 'count': 1, 'unit_value': 0.50, 'subtotal_eur': 0.50},
        ],
        'denomination_counts': {'1c': 1, '50c': 1},
    }

# TC-6.1: Annotated image has same or larger height than input
def test_annotated_image_size(exporter, sample_image, sample_classified_coins):
    annotated = exporter.draw_annotations(sample_image, sample_classified_coins)
    assert annotated.shape[1] == sample_image.shape[1]
    assert annotated.shape[0] >= sample_image.shape[0]

# TC-6.2: Annotated image is modified (not identical to original)
def test_annotation_modifies_image(exporter, sample_image, sample_classified_coins):
    annotated = exporter.draw_annotations(sample_image, sample_classified_coins)
    assert not np.array_equal(annotated, sample_image)

# TC-6.3: Summary banner adds rows above image
def test_summary_banner_adds_height(exporter, sample_image, sample_summary):
    annotated_no_banner = sample_image.copy()
    with_banner = exporter.draw_summary_banner(annotated_no_banner, sample_summary)
    assert with_banner.shape[0] > sample_image.shape[0]

# TC-6.4: Annotated image is saved to disk
def test_save_annotated_image(exporter, sample_image, tmp_path):
    out_path = exporter.save_annotated_image(sample_image, "test_image.jpg")
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0

# TC-6.5: JSON file is created and parseable
def test_json_export_creates_file(exporter, tmp_path, sample_summary, 
                                   sample_classified_coins):
    p5 = {'summary': sample_summary, 'classified_coins': sample_classified_coins}
    json_path = exporter.export_json(p5, "test.jpg")
    assert Path(json_path).exists()
    with open(json_path) as f:
        data = json.load(f)
    assert 'summary' in data
    assert 'coins' in data

# TC-6.6: JSON schema has required keys
def test_json_schema(exporter, sample_summary, sample_classified_coins):
    p5 = {'summary': sample_summary, 'classified_coins': sample_classified_coins}
    json_path = exporter.export_json(p5, "test.jpg")
    with open(json_path) as f:
        data = json.load(f)
    required_top_level = {'run_id', 'source_image', 'timestamp', 'summary', 'coins'}
    assert required_top_level.issubset(data.keys())
    required_coin = {'denomination', 'value_eur', 'confidence', 'circle'}
    for coin in data['coins']:
        assert required_coin.issubset(coin.keys())

# TC-6.7: Total EUR in JSON matches input
def test_json_total_matches(exporter, sample_summary, sample_classified_coins):
    p5 = {'summary': sample_summary, 'classified_coins': sample_classified_coins}
    json_path = exporter.export_json(p5, "test.jpg")
    with open(json_path) as f:
        data = json.load(f)
    assert abs(data['summary']['total_eur'] - 0.51) < 0.001

# TC-6.8: HTML file is created and contains key content
def test_html_report_created(exporter, tmp_path, sample_summary, 
                               sample_classified_coins):
    # Need a real annotated image file for base64 embedding
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = exporter.save_annotated_image(img, "test.jpg")
    p5 = {'summary': sample_summary, 'classified_coins': sample_classified_coins}
    html_path = exporter.generate_html_report(p5, img_path, "test.jpg")
    assert Path(html_path).exists()
    content = Path(html_path).read_text()
    assert '€0.51' in content or 'total' in content.lower()
    assert 'data:image/png;base64,' in content

# TC-6.9: LABEL_MAP covers all denominations
def test_label_map_complete():
    expected_dens = {'1c', '2c', '5c', '10c', '20c', '50c', '1e', '2e', 'UNKNOWN'}
    assert expected_dens.issubset(LABEL_MAP.keys())

# TC-6.10: Confidence color function returns valid BGR tuple
def test_confidence_color(exporter):
    for conf in [0.0, 0.25, 0.50, 0.75, 1.0]:
        color = exporter._confidence_color(conf)
        assert len(color) == 3
        assert all(0 <= c <= 255 for c in color)
```

### Phase 6 Completion Checklist

- [ ] All 10 test cases pass
- [ ] Annotated image visually correct on all 5 sample images
- [ ] JSON schema valid and parseable by Python, Node.js, bash (jq)
- [ ] HTML report renders correctly in Chrome/Firefox
- [ ] Console table prints cleanly in terminal (UTF-8 box characters)
- [ ] All output files created in correct directories
- [ ] Run ID ensures no overwriting between runs
