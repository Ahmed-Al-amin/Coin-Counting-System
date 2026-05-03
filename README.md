# 🪙 Coin Counting System

A classical computer vision system that detects, classifies, and counts coins using Circular Hough Transform and hand-crafted color features. **No machine learning or deep learning** — pure image processing.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run on sample image
python main.py --image data/sample_images/mixed_coins.jpg

# Run with debug output (saves intermediate images)
python main.py --image data/sample_images/mixed_coins.jpg --debug
