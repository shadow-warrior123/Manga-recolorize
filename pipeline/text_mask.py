"""
Text Detection and Masking Module.

Detects English text regions in manga pages, creates binary masks,
and extracts original text pixels for later reinsertion.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def detect_text_regions(
    image: np.ndarray,
    use_cuda: bool = True,
    text_threshold: float = 0.7,
    link_threshold: float = 0.4,
    low_text: float = 0.4,
) -> List[np.ndarray]:
    """
    Detect text regions in a manga page image.
    
    Uses CRAFT (Character Region Awareness for Text Detection) to identify
    text bounding boxes. Falls back to a simple stub if CRAFT unavailable.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV).
        use_cuda: Whether to use CUDA for inference.
        text_threshold: Text confidence threshold.
        link_threshold: Link confidence threshold.
        low_text: Low text bound threshold.
        
    Returns:
        List of polygon arrays representing text region boundaries.
        Each polygon is an array of shape (N, 2) containing (x, y) points.
    """
    logger.info("Detecting text regions in image...")
    
    try:
        from craft_text_detector import Craft
        
        # Initialize CRAFT detector
        craft = Craft(
            output_dir=None,
            crop_type="poly",
            cuda=use_cuda,
            text_threshold=text_threshold,
            link_threshold=link_threshold,
            low_text=low_text,
        )
        
        # Run detection
        prediction_result = craft.detect_text(image)
        
        # Extract polygons from result
        polygons = []
        if prediction_result and "polys" in prediction_result:
            for poly in prediction_result["polys"]:
                if poly is not None and len(poly) > 0:
                    polygons.append(np.array(poly, dtype=np.int32))
        
        # Cleanup
        craft.unload_craftnet_model()
        craft.unload_refinenet_model()
        
        logger.info(f"Detected {len(polygons)} text regions")
        return polygons
        
    except ImportError:
        logger.warning("CRAFT not available, using fallback detection")
        return _fallback_text_detection(image)
    except Exception as e:
        logger.error(f"Error in text detection: {e}")
        return _fallback_text_detection(image)


def _fallback_text_detection(image: np.ndarray) -> List[np.ndarray]:
    """
    Fallback text detection using simple contour analysis.
    
    This is a stub implementation that detects high-contrast regions
    that may contain text. For production, use CRAFT or DBNet.
    
    Args:
        image: Input image as numpy array (BGR format).
        
    Returns:
        List of bounding box polygons.
    """
    logger.info("Using fallback text detection (contour-based)")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to find high-contrast regions
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    
    # Invert for black text on white background (common in manga)
    binary_inv = cv2.bitwise_not(binary)
    
    # Morphological operations to connect text characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(binary_inv, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    h, w = gray.shape[:2]
    min_area = (w * h) * 0.0005  # Minimum 0.05% of image area
    max_area = (w * h) * 0.15   # Maximum 15% of image area
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Filter by aspect ratio (text regions are usually wider)
            aspect_ratio = bw / max(bh, 1)
            if 0.5 < aspect_ratio < 20:
                # Convert to polygon format
                poly = np.array([
                    [x, y],
                    [x + bw, y],
                    [x + bw, y + bh],
                    [x, y + bh]
                ], dtype=np.int32)
                polygons.append(poly)
    
    logger.info(f"Fallback detected {len(polygons)} potential text regions")
    return polygons


def create_text_mask(
    image_shape: Tuple[int, int, int],
    text_regions: List[np.ndarray],
    dilation_size: int = 5,
) -> np.ndarray:
    """
    Create a binary mask from detected text regions.
    
    Args:
        image_shape: Shape of the original image (H, W, C).
        text_regions: List of polygon arrays from detect_text_regions.
        dilation_size: Size of dilation kernel to expand mask slightly.
        
    Returns:
        Binary mask as numpy array (H, W) where 255 = text region, 0 = non-text.
    """
    logger.info("Creating text mask from detected regions...")
    
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Fill polygons
    for poly in text_regions:
        if len(poly) >= 3:
            cv2.fillPoly(mask, [poly], 255)
    
    # Dilate mask slightly to ensure full coverage
    if dilation_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (dilation_size, dilation_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    text_pixel_count = np.sum(mask > 0)
    total_pixels = h * w
    coverage = (text_pixel_count / total_pixels) * 100
    
    logger.info(f"Text mask created: {text_pixel_count} pixels ({coverage:.2f}% coverage)")
    return mask


def extract_text_layer(
    image: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """
    Extract original text pixels as an RGBA layer for later reinsertion.
    
    The alpha channel is set based on the text mask, allowing for
    seamless compositing onto the colorized image.
    
    Args:
        image: Original image as numpy array (BGR format).
        text_mask: Binary mask from create_text_mask.
        
    Returns:
        RGBA image (H, W, 4) containing only the text pixels.
        Non-text pixels have alpha = 0.
    """
    logger.info("Extracting text layer...")
    
    h, w = image.shape[:2]
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    
    # Create RGBA output
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Copy RGB channels
    if len(rgb.shape) == 3:
        rgba[:, :, :3] = rgb
    else:
        rgba[:, :, 0] = rgb
        rgba[:, :, 1] = rgb
        rgba[:, :, 2] = rgb
    
    # Set alpha from mask
    rgba[:, :, 3] = text_mask
    
    text_pixel_count = np.sum(text_mask > 0)
    logger.info(f"Text layer extracted: {text_pixel_count} pixels with content")
    
    return rgba


def save_text_layer(
    text_layer: np.ndarray,
    output_path: str,
) -> str:
    """
    Save text layer to a PNG file with transparency.
    
    Args:
        text_layer: RGBA text layer from extract_text_layer.
        output_path: Path to save the PNG file.
        
    Returns:
        Path to saved file.
    """
    logger.info(f"Saving text layer to: {output_path}")
    
    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PIL Image and save
    pil_image = Image.fromarray(text_layer, mode="RGBA")
    pil_image.save(output_path, "PNG")
    
    return output_path


def process_text_detection(
    image_path: str,
    use_cuda: bool = True,
    save_mask: bool = False,
    save_layer: bool = False,
    output_dir: Optional[str] = None,
    dilation_size: int = 10,  # Increased default for safety
) -> Dict[str, Any]:
    """
    Complete text detection and masking pipeline for a single image.
    
    Args:
        image_path: Path to input image.
        use_cuda: Whether to use CUDA for detection.
        save_mask: Whether to save the binary mask.
        save_layer: Whether to save the text layer.
        output_dir: Directory for saved outputs.
        
    Returns:
        Dictionary containing:
        - 'mask': Binary text mask (H, W)
        - 'text_layer': RGBA text layer (H, W, 4)
        - 'regions': List of detected polygon regions
        - 'mask_path': Path to saved mask (if save_mask=True)
        - 'layer_path': Path to saved layer (if save_layer=True)
    """
    logger.info(f"Processing text detection for: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Detect text regions
    regions = detect_text_regions(image, use_cuda=use_cuda)
    
    # Create mask
    mask = create_text_mask(image.shape, regions, dilation_size=dilation_size)
    
    # Extract text layer
    text_layer = extract_text_layer(image, mask)
    
    result = {
        "mask": mask,
        "text_layer": text_layer,
        "regions": regions,
    }
    
    # Save outputs if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = Path(image_path).stem
        
        if save_mask:
            mask_path = str(output_dir / f"{base_name}_mask.png")
            cv2.imwrite(mask_path, mask)
            result["mask_path"] = mask_path
            logger.info(f"Saved mask to: {mask_path}")
        
        if save_layer:
            layer_path = str(output_dir / f"{base_name}_text_layer.png")
            save_text_layer(text_layer, layer_path)
            result["layer_path"] = layer_path
    
    return result


if __name__ == "__main__":
    # Simple test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        result = process_text_detection(
            test_image,
            use_cuda=False,
            save_mask=True,
            save_layer=True,
            output_dir="./test_output"
        )
        print(f"Detected {len(result['regions'])} text regions")
    else:
        print("Usage: python text_mask.py <image_path>")
