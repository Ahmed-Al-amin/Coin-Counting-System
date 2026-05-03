# 🪙 Coin Counting System (Classical Computer Vision)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated system to detect, classify, and count the total value of Euro coins in a single photograph using **Classical Computer Vision** techniques. No deep learning, no neural networks—just pure geometry and color analysis.

---

## 🚀 Overview

The **Coin Counting System** implements a robust 6-phase pipeline to transform a raw image into a detailed monetary report. It specifically targets Euro currency, identifying denominations from 1 cent to 2 Euros by analyzing their physical dimensions and color profiles.

### Key Features
- **Deterministic Pipeline:** Every stage is inspectable and explainable.
- **Circle Detection:** High-precision Circular Hough Transform (CHT) with Non-Maximum Suppression (NMS).
- **Rule-Based Classification:** A two-stage decision tree using size ratios and HSV/LAB color statistics.
- **Comprehensive Output:** Annotated images, JSON data, console tables, and interactive HTML reports.
- **Debug Mode:** Save intermediate results (edge maps, ROI crops, feature plots) to understand the "why" behind every detection.

---

## 🛠️ System Architecture

The project is structured as a modular pipeline:

1.  **Preprocessing:** Grayscale, CLAHE contrast enhancement, Gaussian blur, and Canny edge detection.
2.  **Detection:** Circular Hough Transform to locate coins; NMS to handle overlaps.
3.  **Segmentation:** ROI extraction with circular masking and 128x128 normalization.
4.  **Feature Extraction:** Calculation of relative size ratios, mean HSV/LAB values, and hue histograms.
5.  **Classification:** Logic-based assignment of denominations (Copper, Gold, Bicolor).
6.  **Output:** Visualization and data export.

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- `pip` (Python package installer)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ahmed-Al-amin/Coin-Counting-System.git
   cd Coin-Counting-System
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🖥️ Usage

### Basic Execution
Process an image and see the results in the console:
```bash
python main.py --image data/sample_images/mixed_coins.jpg
```

### Debug Mode
Generate intermediate images for every phase (saved in `outputs/intermediate/`):
```bash
python main.py --image data/sample_images/mixed_coins.jpg --debug
```

### Custom Configuration
Adjust detection sensitivity or classification thresholds in `config.yaml`:
```bash
python main.py --image data/sample_images/mixed_coins.jpg --config config.yaml
```

---

## 📊 Supported Coins

| Coin | Value | Color Group |
| :--- | :--- | :--- |
| **1, 2, 5 cent** | €0.01 - 0.05 | Copper |
| **10, 20, 50 cent** | €0.10 - 0.50 | Gold |
| **1, 2 Euro** | €1.00 - 2.00 | Bicolor (Silver/Gold) |

---

## 📂 Project Structure

- `src/`: Core logic (one module per pipeline phase).
- `data/`: Sample images and ground truth annotations.
- `docs/`: Detailed documentation for each implementation phase.
- `outputs/`: Generated annotations, logs, and reports.
- `tests/`: Comprehensive pytest suite for unit and integration testing.

---

## 🧪 Testing

Run the full test suite to verify the pipeline:
```bash
pytest tests/test_integration.py -v
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` (if available) or the PRD for more information.

## 👥 Contributors
- **Ahmed Al-amin** - [GitHub](https://github.com/Ahmed-Al-amin)
