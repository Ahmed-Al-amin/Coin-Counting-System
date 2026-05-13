"""Phase 6_output module - Coin Counting System."""

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

class ResultExporter:
    def __init__(self, config: dict, output_dir: str = "outputs"):
        self.config = config
        self.output_dir = output_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def draw_annotations(self, image: np.ndarray, 
                          classified_coins: List[Dict]) -> np.ndarray:
        """
        Draw annotated overlays on image:
        - Colored circle outline (color = coin color group)
        - Label with size category (e.g., "Medium") above coin
        - Confidence bar below coin (green=high, red=low)
        """
        annotated = image.copy()
        
        for coin in classified_coins:
            if not coin.get('is_coin', True):
                continue

            x, y, r = coin['circle']
            cat = coin['category']
            conf = coin['confidence']
            cgroup = coin['color_group']
            
            color = COLORS.get(cgroup, COLORS['UNKNOWN'])
            
            # Outer circle
            thickness = max(2, r // 20)
            cv2.circle(annotated, (x, y), r, color, thickness)
            
            # Center dot
            cv2.circle(annotated, (x, y), 3, color, -1)
            
            # Category label
            label = cat
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.4, r / 80)
            font_thick = max(1, int(r / 40))
            
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
            
            # Filled portion
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
        
        # Total count (large text)
        total_text = f"Total Coins: {summary['total_count']}"
        cv2.putText(banner, total_text, (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 230, 100), 2, cv2.LINE_AA)
        
        # Breakdown
        breakdown_text = " | ".join([f"{b['category']}: {b['count']}" for b in summary['breakdown']])
        cv2.putText(banner, breakdown_text, (w - 450, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        
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
        """Export full results to JSON."""
        results = {
            "run_id":       self.run_id,
            "source_image": str(source_path),
            "timestamp":    datetime.now().isoformat(),
            "summary":      phase5_result['summary'],
            "coins": [
                {
                    "coin_index":   c['coin_index'],
                    "category":     c['category'],
                    "is_coin":      c['is_coin'],
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
        lines.append("\n" + "═" * 44)
        lines.append("  🪙  GENERIC COIN COUNTING RESULTS")
        lines.append("═" * 44)
        lines.append(f"  {'Category':<15} {'Count':>10} {'% of Total':>15}")
        lines.append("─" * 44)
        
        total = summary['total_count']
        for row in breakdown:
            pct = (row['count'] / total * 100) if total > 0 else 0
            lines.append(f"  {row['category']:<15} {row['count']:>10} {pct:>14.1f}%")
        
        lines.append("─" * 44)
        lines.append(f"  {'TOTAL COINS':<15} {total:>10}")
        lines.append("═" * 44)
        lines.append(f"  Average confidence: {summary['avg_confidence']:.1%}")
        if summary['low_confidence_count'] > 0:
            lines.append(f"  ⚠️  Low-confidence/Noise: {summary['low_confidence_count']}")
        lines.append("")
        
        output = "\n".join(lines)
        print(output)
        return output

    def generate_html_report(self, phase5_result: dict,
                              annotated_path: str,
                              source_path: str) -> str:
        """Generate a self-contained HTML report."""
        summary = phase5_result['summary']
        coins = phase5_result['classified_coins']
        
        with open(annotated_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        rows = ""
        for coin in coins:
            status = "COIN" if coin['is_coin'] else "NOISE"
            conf = f"{coin['confidence']:.0%}"
            conf_color = '#2ecc71' if coin['confidence'] >= 0.75 else \
                         '#f39c12' if coin['confidence'] >= 0.50 else '#e74c3c'
            rows += f"""
            <tr>
              <td>{coin['coin_index']}</td>
              <td><strong>{coin['category']}</strong></td>
              <td>{status}</td>
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
  <h1>🪙 Generic Coin Counting Report</h1>
  <p style="text-align:center;color:#aaa">Source: {Path(source_path).name} | Run: {self.run_id}</p>
  <div class="summary">
    <div class="card">
      <div class="value">{summary['total_count']}</div>
      <div class="label">Total Coins</div>
    </div>
    <div class="card">
      <div class="value">{summary['avg_confidence']:.0%}</div>
      <div class="label">Avg Confidence</div>
    </div>
  </div>
  <img src="data:image/png;base64,{img_b64}" alt="Annotated coin image">
  <h2>Per-Detection Results</h2>
  <table>
    <tr><th>#</th><th>Category</th><th>Status</th><th>Confidence</th><th>Color Group</th><th>Reasoning</th></tr>
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
