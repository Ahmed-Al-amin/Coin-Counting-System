"""Generate synthetic sample images for testing."""

import cv2
import numpy as np
from pathlib import Path

def create_single_coin_image(output_path: str, color: tuple, radius: int, 
                              position: tuple, background: tuple = (240, 240, 240)):
    """Create a single coin image."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = background
    cv2.circle(img, position, radius, color, -1)
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

def create_mixed_coins_image(output_path: str):
    """Create image with multiple coins."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    
    # 1c (copper, small)
    cv2.circle(img, (150, 150), 45, (50, 100, 200), -1)
    # 10c (gold, medium)
    cv2.circle(img, (350, 150), 50, (50, 165, 210), -1)
    # 50c (gold, large)
    cv2.circle(img, (550, 150), 60, (50, 165, 210), -1)
    # €1 (bicolor, largest)
    cv2.circle(img, (400, 400), 65, (180, 140, 220), -1)
    
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

def create_touching_coins(output_path: str):
    """Create image with touching coins."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)
    
    cv2.circle(img, (200, 250), 60, (50, 100, 200), -1)
    cv2.circle(img, (270, 250), 60, (50, 165, 210), -1)
    
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

def create_low_light_image(output_path: str):
    """Create low-light simulation."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:] = (80, 80, 80)
    cv2.circle(img, (250, 250), 60, (50, 100, 200), -1)
    cv2.imwrite(output_path, img)
    print(f"  Created: {output_path}")

if __name__ == "__main__":
    sample_dir = Path(__file__).parent / "sample_images"
    sample_dir.mkdir(exist_ok=True)
    
    print("Generating sample images...")
    create_single_coin_image(str(sample_dir / "single_coin.jpg"), (50, 100, 200), 60, (250, 250))
    create_mixed_coins_image(str(sample_dir / "mixed_coins.jpg"))
    create_touching_coins(str(sample_dir / "touching_coins.jpg"))
    create_low_light_image(str(sample_dir / "low_light.jpg"))
    print("✓ Sample images created")
