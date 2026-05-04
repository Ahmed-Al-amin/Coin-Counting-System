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
         'reasoning': ['Color group: GOLD', 'Large gold -> 50c']},
        {'coin_index': 1, 'circle': (250, 200, 30), 'denomination': '1c',
         'value_eur': 0.01, 'confidence': 0.72, 'color_group': 'COPPER',
         'reasoning': ['Color group: COPPER', 'Small -> 1c']},
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
def test_json_export_creates_file(exporter, tmp_path, sample_summary, sample_classified_coins):
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
def test_html_report_created(exporter, tmp_path, sample_summary, sample_classified_coins):
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