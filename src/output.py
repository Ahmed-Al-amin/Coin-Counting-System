"""Phase 6_output module - Coin Counting System."""

import cv2
import numpy as np
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
        """Return BGR color: green (high_conf) -> yellow -> red (low_conf)."""
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

    def run(self, all_phase_results: dict, source_path: str) -> dict:
        """Execute output generation (annotations and console table)."""
        p1 = all_phase_results['phase1']
        p5 = all_phase_results['phase5']
        
        # Annotate image
        annotated = self.draw_annotations(p1['scaled'], p5['classified_coins'])
        annotated = self.draw_summary_banner(annotated, p5['summary'])
        
        # Save annotated image
        img_path = self.save_annotated_image(annotated, source_path)
        
        # Print table to console
        self.print_table(p5)
        
        return {
            'annotated_image': annotated,
            'annotated_path':  img_path,
        }
