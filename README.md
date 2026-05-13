# 🪙 Generic Coin Counting System (Classical Computer Vision)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

An automated system to detect, validate, and count coins in a single photograph using **Classical Computer Vision** techniques. No deep learning, no neural networks—just pure geometry and color analysis.

---

## 🚀 Overview

The **Generic Coin Counting System** implements a robust 6-phase pipeline to transform a raw image into a detailed counting report. It identifies circular objects, validates them as coins using surface features, and categorizes them into **Small, Medium, and Large** groups based on their physical dimensions relative to the image size.

### Key Features
- **Deterministic Pipeline:** Every stage is inspectable and explainable.
- **Circle Detection:** High-precision Circular Hough Transform (CHT) with **Contour-based Fallback** for robust detection in all conditions.
- **Heuristic Validation:** A robust validator using color saturation and hue entropy to filter noise.
- **Generic Categorization:** Size-based grouping (Small, Medium, Large) applicable to any currency.
- **Comprehensive Output:** Annotated images, JSON data, console tables, and interactive HTML reports.
- **Debug Mode:** Save intermediate results (edge maps, ROI crops, feature plots) to understand the "why" behind every detection.

---

## 🛠️ System Architecture

The project is structured as a modular pipeline:

1.  **Preprocessing:** Grayscale, CLAHE contrast enhancement, Gaussian blur, and Canny edge detection.
2.  **Detection:** Circular Hough Transform to locate coins; **Contour detection fallback**; NMS to handle overlaps.
3.  **Segmentation:** ROI extraction with circular masking and 128x128 normalization.
4.  **Feature Extraction:** Calculation of relative size ratios, mean HSV/LAB values, and hue histogram entropy.
5.  **Validation & Categorization:** Generic validator to confirm "coin-ness" and group by size.
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

2. **Install dependencies:**
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
Adjust detection sensitivity or categorization thresholds in `config.yaml`:
```bash
python main.py --image data/sample_images/mixed_coins.jpg --config config.yaml
```

---

## 📊 Size Categorization

| Category | Relative Size (r/diagonal) |
| :--- | :--- |
| **Small** | < 0.035 |
| **Medium** | 0.035 - 0.055 |
| **Large** | > 0.055 |

---

## 📂 Project Structure

- `src/`: Core logic (one module per pipeline phase).
- `data/`: Sample images.
- `docs/`: Detailed documentation for each implementation phase.
- `outputs/`: Generated annotations, logs, and reports.
- `tests/`: Comprehensive pytest suite for unit and integration testing.

---

## 🧪 Testing

Run the full test suite to verify the pipeline:
```bash
python -m pytest tests/ -v
```

---

## 📜 License
Distributed under the MIT License.

## 👥 Contributors
- **Ahmed Al-amin** - [GitHub](https://github.com/Ahmed-Al-amin)
