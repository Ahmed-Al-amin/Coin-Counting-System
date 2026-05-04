
import cv2
import numpy as np
import json
import os
import base64
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
        self.config = config
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
            den = coin['denomination']
            conf = coin['confidence']
            cgroup = coin['color_group']
            
            color = COLORS.get(cgroup, COLORS['UNKNOWN'])
            
            # Outer circle
            thickness = max(2, r // 20)
            cv2.circle(annotated, (x, y), r, color, thickness)
            
            # Center dot
            cv2.circle(annotated, (x, y), 3, color, -1)
            
            # Denomination label
            label = LABEL_MAP.get(den, '?')
            font = cv2.FONT_HERSHEY_DUPLEX
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
            
            # Confidence bar
            bar_w = int(r * 1.5)
            bar_h = max(4, r // 10)
            bar_x = x - bar_w // 2
            bar_y = y + r + 6
            
            # Background (dark gray)
            cv2.rectangle(annotated, (bar_x, bar_y), 
                          (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            
            # Filled portion (green -> yellow -> red by confidence)
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
        """Return BGR color: green (high conf) -> yellow -> red (low conf)."""
        if conf >= 0.75:
            return (0, 200, 0)     # Green
        elif conf >= 0.50:
            return (0, 200, 200)   # Yellow
        else:
            return (0, 0, 200)     # Red

    def draw_summary_banner(self, annotated: np.ndarray, 
                             summary: Dict) -> np.ndarray:
        """
        Draw a summary banner at the top of the annotated image.
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

    def save_annotated_image(self, annotated: np.ndarray, 
                              source_path: str) -> str:
        """Save annotated result image to outputs/annotated/."""
        stem = Path(source_path).stem
        out_path = f"{self.output_dir}/annotated/{stem}_{self.run_id}.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated)
        return out_path

    def export_json(self, phase5_result: dict, 
                    source_path: str) -> str:
        """Export full results to JSON for programmatic use."""
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
        
        stem = Path(source_path).stem
        out_path = f"{self.output_dir}/reports/{stem}_{self.run_id}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return out_path

    def print_table(self, phase5_result: dict) -> str:
        """Print and return a formatted console summary table."""
        summary = phase5_result['summary']
        breakdown = summary['breakdown']
        
        lines = []
        lines.append("\n" + "═" * 54)
        lines.append("  💰  COIN COUNTING RESULTS")
        lines.append("═" * 54)
        lines.append(f"  {'Denomination':<14} {'Count':>6} {'Unit':>8} {'Subtotal':>10}")
        lines.append("─" * 54)
        
        for row in breakdown:
            den_sym = LABEL_MAP.get(row['denomination'], row['denomination'])
            count = row['count']
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

    def generate_html_report(self, phase5_result: dict,
                              annotated_path: str,
                              source_path: str) -> str:
        """Generate a self-contained HTML report with embedded image."""
        summary = phase5_result['summary']
        coins = phase5_result['classified_coins']
        
        with open(annotated_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        rows = ""
        for coin in coins:
            den = LABEL_MAP.get(coin['denomination'], '?')
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
    body {{ font-family: 'Segoe UI', sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #eee; }}
    h1 {{ color: #f0c040; text-align: center; letter-spacing: 2px; }}
    .summary {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; }}
    .card {{ background: #16213e; border-radius: 12px; padding: 20px 30px; text-align: center; border: 1px solid #0f3460; }}
    .card .value {{ font-size: 2em; color: #f0c040; font-weight: bold; }}
    .card .label {{ color: #aaa; font-size: 0.9em; }}
    img {{ max-width: 100%; border-radius: 8px; border: 2px solid #0f3460; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
    th {{ background: #0f3460; padding: 10px; text-align: left; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #333; }}
    tr:hover {{ background: #16213e; }}
  </style>
</head>
<body>
  <h1>🪙 Coin Counting Report</h1>
  <p style="text-align:center;color:#aaa">Source: {Path(source_path).name} | Run: {self.run_id}</p>
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
    <tr><th>#</th><th>Denomination</th><th>Value</th><th>Confidence</th><th>Color Group</th><th>Reasoning</th></tr>
    {rows}
  </table>
</body>
</html>"""
        
        stem = Path(source_path).stem
        out_path = f"{self.output_dir}/reports/{stem}_{self.run_id}.html"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return out_path

    def run(self, all_phase_results: dict, source_path: str) -> dict:
        """Execute all output generation."""
        p1 = all_phase_results['phase1']
        p5 = all_phase_results['phase5']
        
        # Annotate image
        annotated = self.draw_annotations(p1['scaled'], p5['classified_coins'])
        annotated = self.draw_summary_banner(annotated, p5['summary'])
        
        # Save outputs
        img_path = self.save_annotated_image(annotated, source_path)
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