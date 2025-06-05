import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt

def create_output_dir(output_path: str) -> None:
    """Create output directory if it doesn't exist."""
    Path(output_path).mkdir(parents=True, exist_ok=True)

def load_image(image_path: str) -> np.ndarray:
    """Load image from path."""
    return cv2.imread(image_path)

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """Apply histogram equalization to enhance contrast."""
    return cv2.equalizeHist(image)

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8,8)) -> np.ndarray:
    """Enhance contrast using CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def gaussian_smoothing(image: np.ndarray, kernel_size: Tuple[int, int] = (5,5)) -> np.ndarray:
    """Apply Gaussian smoothing to reduce noise."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def process_single_image(image_path: str, output_path: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Process a single image with all preprocessing steps."""
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    image = convert_to_grayscale(image)

    # Apply preprocessing steps
    image = histogram_equalization(image)
    image = enhance_contrast(image)
    image = gaussian_smoothing(image)
    image = resize_image(image, target_size)

    # Save processed image
    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, image)

def process_all_images(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (224, 224)) -> None:
    """Process all images in the input directory."""
    create_output_dir(output_dir)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)
            process_single_image(input_path, output_dir, target_size)

def visualize_preprocessing(image_path: str) -> None:
    """Visualize preprocessing steps on a single image."""
    # Load original image
    original = load_image(image_path)
    if original is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to grayscale
    original = convert_to_grayscale(original)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')

    # Histogram equalization
    hist_eq = histogram_equalization(original)
    axes[0,1].imshow(hist_eq, cmap='gray')
    axes[0,1].set_title('Histogram Equalization')
    axes[0,1].axis('off')

    # Contrast enhancement
    contrast = enhance_contrast(original)
    axes[1,0].imshow(contrast, cmap='gray')
    axes[1,0].set_title('Contrast Enhancement')
    axes[1,0].axis('off')

    # Gaussian smoothing
    smoothed = gaussian_smoothing(original)
    axes[1,1].imshow(smoothed, cmap='gray')
    axes[1,1].set_title('Gaussian Smoothing')
    axes[1,1].axis('off')

    plt.tight_layout()
    plt.show() 