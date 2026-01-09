"""
Line Art Extraction Module.

Converts manga pages to clean line art suitable for ControlNet conditioning.
Removes text regions using the provided mask.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_lineart(
    image: np.ndarray,
    method: str = "canny",
    invert: bool = True,
    blur_size: int = 5,
    threshold1: int = 50,
    threshold2: int = 150,
) -> np.ndarray:
    """
    Extract clean line art from a manga page image.
    
    Supports multiple extraction methods for different manga styles.
    
    Args:
        image: Input image as numpy array (BGR or grayscale).
        method: Extraction method - 'canny', 'threshold', or 'adaptive'.
        invert: If True, output white lines on black background.
        blur_size: Gaussian blur kernel size for noise reduction.
        threshold1: Lower threshold for Canny edge detection.
        threshold2: Upper threshold for Canny edge detection.
        
    Returns:
        Line art image as numpy array (grayscale, H, W).
    """
    logger.info(f"Extracting line art using method: {method}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply Gaussian blur to reduce noise
    if blur_size > 0:
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    else:
        blurred = gray
    
    if method == "canny":
        lineart = _extract_canny(blurred, threshold1, threshold2)
    elif method == "threshold":
        lineart = _extract_threshold(blurred)
    elif method == "adaptive":
        lineart = _extract_adaptive(blurred)
    elif method == "xdog":
        lineart = _extract_xdog(blurred)
    else:
        logger.warning(f"Unknown method '{method}', falling back to canny")
        lineart = _extract_canny(blurred, threshold1, threshold2)
    
    # Invert if requested (white lines on black -> black lines on white)
    if not invert:
        lineart = cv2.bitwise_not(lineart)
    
    logger.info(f"Line art extracted: shape={lineart.shape}")
    return lineart


def _extract_canny(gray: np.ndarray, threshold1: int, threshold2: int) -> np.ndarray:
    """Extract edges using Canny edge detection."""
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Dilate slightly to connect broken lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    return edges


def _extract_threshold(gray: np.ndarray) -> np.ndarray:
    """Extract lines using simple thresholding (for clean manga)."""
    # Manga typically has clean black lines on white
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Invert to get white lines on black
    return cv2.bitwise_not(binary)


def _extract_adaptive(gray: np.ndarray) -> np.ndarray:
    """Extract lines using adaptive thresholding (handles varying backgrounds)."""
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )
    
    # Invert to get white lines on black
    return cv2.bitwise_not(binary)


def _extract_xdog(gray: np.ndarray, sigma: float = 0.5, k: float = 1.6, p: float = 20) -> np.ndarray:
    """
    Extract lines using XDoG (eXtended Difference of Gaussians).
    
    Produces more artistic, stylized line art similar to manga.
    
    Args:
        gray: Grayscale input image.
        sigma: Standard deviation for first Gaussian.
        k: Multiplier for second Gaussian sigma.
        p: Sharpening parameter.
    """
    # First Gaussian
    g1 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    
    # Second Gaussian with larger sigma
    g2 = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma * k)
    
    # Difference of Gaussians
    dog = g1 - g2
    
    # Apply sharpening
    dog = dog * (1 + p)
    
    # Threshold
    dog = np.where(dog >= 0, 255, 0).astype(np.uint8)
    
    return dog


def remove_text_regions(
    lineart: np.ndarray,
    text_mask: np.ndarray,
    inpaint: bool = True,
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    Remove text regions from line art using the provided mask.
    
    Args:
        lineart: Line art image (grayscale or RGB).
        text_mask: Binary mask where 255 = text regions.
        inpaint: If True, inpaint the masked regions. If False, fill with black.
        inpaint_radius: Radius for inpainting algorithm.
        
    Returns:
        Line art with text regions removed/inpainted.
    """
    logger.info("Removing text regions from line art...")
    
    # Ensure mask matches lineart dimensions
    if text_mask.shape[:2] != lineart.shape[:2]:
        text_mask = cv2.resize(
            text_mask, 
            (lineart.shape[1], lineart.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    if inpaint:
        # Use OpenCV inpainting to fill text regions
        if len(lineart.shape) == 2:
            # Grayscale: convert to 3-channel for inpainting
            lineart_3ch = cv2.cvtColor(lineart, cv2.COLOR_GRAY2BGR)
            result = cv2.inpaint(lineart_3ch, text_mask, inpaint_radius, cv2.INPAINT_TELEA)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            result = cv2.inpaint(lineart, text_mask, inpaint_radius, cv2.INPAINT_TELEA)
    else:
        # Simple fill with black (or white depending on lineart style)
        result = lineart.copy()
        result[text_mask > 0] = 0
    
    removed_pixels = np.sum(text_mask > 0)
    logger.info(f"Removed text from {removed_pixels} pixels")
    
    return result


def prepare_for_controlnet(
    lineart: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    ensure_rgb: bool = True,
) -> np.ndarray:
    """
    Prepare line art image for ControlNet conditioning.
    
    ControlNet typically expects:
    - RGB image (3 channels)
    - Specific size (often 512x512 or 1024x1024 for SDXL)
    - Normalized or uint8 values
    
    Args:
        lineart: Line art image (grayscale or RGB).
        target_size: Optional (width, height) to resize to.
        ensure_rgb: If True, convert grayscale to RGB.
        
    Returns:
        Line art prepared for ControlNet input.
    """
    logger.info("Preparing line art for ControlNet...")
    
    result = lineart.copy()
    
    # Resize if needed
    if target_size is not None:
        result = cv2.resize(result, target_size, interpolation=cv2.INTER_LANCZOS4)
        logger.info(f"Resized to: {target_size}")
    
    # Convert to RGB if needed
    if ensure_rgb and len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    elif ensure_rgb and result.shape[2] == 4:
        result = cv2.cvtColor(result, cv2.COLOR_BGRA2RGB)
    elif ensure_rgb and result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    logger.info(f"ControlNet input prepared: shape={result.shape}")
    return result


def process_lineart_extraction(
    image_path: str,
    text_mask: Optional[np.ndarray] = None,
    method: str = "adaptive",
    target_size: Optional[Tuple[int, int]] = None,
    save_output: bool = False,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Complete line art extraction pipeline.
    
    Args:
        image_path: Path to input image.
        text_mask: Optional binary mask for text removal.
        method: Line art extraction method.
        target_size: Optional target size for ControlNet.
        save_output: Whether to save the result.
        output_path: Path for saved output.
        
    Returns:
        Processed line art ready for ControlNet.
    """
    logger.info(f"Processing line art extraction for: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Extract line art
    lineart = extract_lineart(image, method=method)
    
    # Remove text if mask provided
    if text_mask is not None:
        lineart = remove_text_regions(lineart, text_mask)
    
    # Prepare for ControlNet
    result = prepare_for_controlnet(lineart, target_size)
    
    # Save if requested
    if save_output and output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved line art to: {output_path}")
    
    return result


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = process_lineart_extraction(
            test_image,
            method="adaptive",
            save_output=True,
            output_path="./test_output/lineart.png"
        )
        print(f"Line art extracted: {result.shape}")
    else:
        print("Usage: python lineart.py <image_path>")
