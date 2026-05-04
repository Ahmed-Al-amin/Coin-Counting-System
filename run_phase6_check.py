# run_phase6_check.py
import cv2
import numpy as np
from src.phase6_output import ResultExporter

# 1. Mock Phase 1 Image (Dark background)
image = np.full((600, 600, 3), (30, 30, 30), dtype=np.uint8)

# 2. Mock Phase 5 Classification Results (Variety of coins)
mock_classified_coins = [
    {'coin_index': 0, 'circle': (150, 200, 80), 'denomination': '2e', 
     'value_eur': 2.00, 'confidence': 0.95, 'color_group': 'BICOLOR', 
     'reasoning': ['Large', 'Bicolor -> €2']},
    {'coin_index': 1, 'circle': (450, 200, 60), 'denomination': '50c', 
     'value_eur': 0.50, 'confidence': 0.82, 'color_group': 'GOLD', 
     'reasoning': ['Gold', 'Medium -> 50c']},
    {'coin_index': 2, 'circle': (300, 450, 45), 'denomination': '10c', 
     'value_eur': 0.10, 'confidence': 0.45, 'color_group': 'GOLD', 
     'reasoning': ['Low conf', 'Scratched -> 10c']}, # Low confidence for testing red bar
    {'coin_index': 3, 'circle': (100, 450, 35), 'denomination': '1c', 
     'value_eur': 0.01, 'confidence': 0.88, 'color_group': 'COPPER', 
     'reasoning': ['Small', 'Copper -> 1c']}
]

mock_summary = {
    'total_eur': 2.61, 'total_coins': 4, 'avg_confidence': 0.775,
    'low_confidence_count': 1,
    'breakdown': [
        {'denomination': '2e',  'count': 1, 'unit_value': 2.00, 'subtotal_eur': 2.00},
        {'denomination': '50c', 'count': 1, 'unit_value': 0.50, 'subtotal_eur': 0.50},
        {'denomination': '10c', 'count': 1, 'unit_value': 0.10, 'subtotal_eur': 0.10},
        {'denomination': '1c',  'count': 1, 'unit_value': 0.01, 'subtotal_eur': 0.01},
    ]
}

all_phase_results = {
    'phase1': {'scaled': image},
    'phase2': {}, 'phase3': {}, 'phase4': {}, 
    'phase5': {'classified_coins': mock_classified_coins, 'summary': mock_summary}
}

# 3. Run the Exporter
exporter = ResultExporter(config={'generate_html': True})
results = exporter.run(all_phase_results, source_path="sample_image_1.jpg")

print("\n🚀 Export Complete!")
print(f"📁 Check your 'outputs' folder.")