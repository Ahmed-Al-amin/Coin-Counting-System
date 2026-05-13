#!#/usr/bin/env python3
"""
Coin Counting System - Main Entry Point
Usage: python main.py --image path/to/coins.jpg [--debug] [--config config.yaml]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
    import cv2
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Run: pip install -r requirements.txt")
    sys.exit(1)

# Import phase modules
from src.preprocessing import Preprocessor
from src.detection import CoinDetector
from src.segmentation import CoinSegmenter
from src.features import FeatureExtractor
from src.classification import CoinClassifier
from src.output import ResultExporter
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Coin Counting System')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--config', '-c', default='config.yaml', help='Config file path')
    parser.add_argument('--debug', '-d', action='store_true', help='Save intermediate outputs')
    parser.add_argument('--output-dir', '-o', default='outputs', help='Output directory')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config['debug'] = args.debug
    config['output_dir'] = args.output_dir
    
    # Setup logger
    log_dir = Path(args.output_dir) / "logs"
    logger = setup_logger(str(log_dir))
    
    logger.info(f"Processing image: {args.image}")
    logger.info(f"Using config: {args.config}")
    start_time = time.time()
    
    try:
        # Phase 1: Preprocessing
        logger.info("Starting Phase 1: Preprocessing")
        preprocessor = Preprocessor(config.get('preprocessing', {}), debug=args.debug)
        phase1_result = preprocessor.run(args.image)
        
        # Phase 2: Detection
        logger.info("Starting Phase 2: Detection")
        detector = CoinDetector(config.get('detection', {}), debug=args.debug)
        phase2_result = detector.run(phase1_result)
        
        if phase2_result.get('count', 0) == 0:
            logger.warning("No coins detected!")
            return
        
        logger.info(f"Found {phase2_result['count']} coins")
        
        # Phase 3: Segmentation
        logger.info("Starting Phase 3: Segmentation")
        segmenter = CoinSegmenter(config.get('segmentation', {}), debug=args.debug)
        phase3_result = segmenter.run(phase1_result, phase2_result)
        
        # Phase 4: Features
        logger.info("Starting Phase 4: Feature Extraction")
        extractor = FeatureExtractor(config.get('features', {}), debug=args.debug)
        phase4_result = extractor.run(phase1_result, phase2_result, phase3_result)
        
        # Phase 5: Classification
        logger.info("Starting Phase 5: Classification")
        classifier = CoinClassifier(config.get('classification', {}))
        phase5_result = classifier.run(phase4_result)
        
        # Phase 6: Output
        logger.info("Starting Phase 6: Output Generation")
        exporter = ResultExporter(config.get('output', {}), output_dir=args.output_dir)
        all_results = {
            'phase1': phase1_result,
            'phase2': phase2_result,
            'phase3': phase3_result,
            'phase4': phase4_result,
            'phase5': phase5_result,
        }
        phase6_result = exporter.run(all_results, args.image)
        
        elapsed = time.time() - start_time
        logger.success(f"Pipeline complete in {elapsed:.2f}s")
        logger.info(f"Total coins detected: {phase5_result['summary']['total_count']}")
        logger.info(f"Outputs saved to: {args.output_dir}/")
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
