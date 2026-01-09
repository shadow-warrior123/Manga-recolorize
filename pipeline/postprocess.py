"""
Post-Processing Module.

Handles text reinsertion, upscaling, and edge preservation
for the final colorized output.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def reinsert_text(
    colorized: np.ndarray,
    text_layer: np.ndarray,
    blend_mode: str = "over",
) -> np.ndarray:
    """
    Reinsert original text pixels onto the colorized image.
    
    Uses the alpha channel from the text layer for compositing.
    
    Args:
        colorized: Colorized image as numpy array (RGB).
        text_layer: RGBA text layer from text_mask.extract_text_layer.
        blend_mode: Blending mode - 'over' (default) or 'replace'.
        
    Returns:
        Final image with text reinserted (RGB).
    """
    logger.info("Reinserting text layer...")
    
    # Ensure same size
    if text_layer.shape[:2] != colorized.shape[:2]:
        text_layer = cv2.resize(
            text_layer,
            (colorized.shape[1], colorized.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Extract alpha channel
    alpha = text_layer[:, :, 3].astype(np.float32) / 255.0
    text_rgb = text_layer[:, :, :3]
    
    # Ensure colorized is RGB
    if len(colorized.shape) == 2:
        colorized = cv2.cvtColor(colorized, cv2.COLOR_GRAY2RGB)
    
    # Composite based on blend mode
    if blend_mode == "over":
        # Standard alpha compositing
        result = colorized.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c] = (
                text_rgb[:, :, c] * alpha + 
                result[:, :, c] * (1 - alpha)
            )
        result = np.clip(result, 0, 255).astype(np.uint8)
    elif blend_mode == "replace":
        # Direct replacement where alpha > 0
        result = colorized.copy()
        mask = alpha > 0.5
        for c in range(3):
            result[:, :, c][mask] = text_rgb[:, :, c][mask]
    else:
        logger.warning(f"Unknown blend mode '{blend_mode}', using 'over'")
        result = reinsert_text(colorized, text_layer, "over")
    
    text_pixels = np.sum(alpha > 0)
    logger.info(f"Reinserted {text_pixels} text pixels")
    
    return result


def upscale_image(
    image: np.ndarray,
    scale: int = 4,
    model_name: str = "RealESRGAN_x4plus_anime_6B",
    denoise_strength: float = 0.5,
    tile: int = 256,
    tile_pad: int = 10,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Upscale image using Real-ESRGAN anime model.
    
    Args:
        image: Input image as numpy array (RGB).
        scale: Upscaling factor (2 or 4).
        model_name: Real-ESRGAN model name.
        denoise_strength: Denoising strength (0-1).
        tile: Tile size for processing large images.
        tile_pad: Padding for tiles to avoid seams.
        use_gpu: Whether to use GPU for upscaling.
        
    Returns:
        Upscaled image as numpy array (RGB).
    """
    logger.info(f"Upscaling image by {scale}x using {model_name}...")
    
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        # Select model based on name
        if model_name == "RealESRGAN_x4plus_anime_6B":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=6,
                num_grow_ch=32,
                scale=4
            )
            netscale = 4
            model_path = None  # Will be downloaded
        elif model_name == "RealESRGAN_x4plus":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )
            netscale = 4
            model_path = None
        else:
            logger.warning(f"Unknown model {model_name}, using anime model")
            return upscale_image(image, scale, "RealESRGAN_x4plus_anime_6B")
        
        # Initialize upsampler
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            model=model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=0,
            half=use_gpu,  # FP16 on GPU
            gpu_id=0 if use_gpu else None,
        )
        
        # Upscale
        output, _ = upsampler.enhance(image, outscale=scale)
        
        logger.info(f"Upscaled: {image.shape} -> {output.shape}")
        return output
        
    except ImportError as e:
        logger.warning(f"Real-ESRGAN not available: {e}")
        return _fallback_upscale(image, scale)
    except Exception as e:
        logger.error(f"Upscaling failed: {e}")
        return _fallback_upscale(image, scale)


def _fallback_upscale(
    image: np.ndarray,
    scale: int,
) -> np.ndarray:
    """
    Fallback upscaling using Lanczos interpolation.
    
    # TODO: Implement more sophisticated CPU-based upscaling
    
    Args:
        image: Input image.
        scale: Upscaling factor.
        
    Returns:
        Upscaled image.
    """
    logger.info("Using fallback upscaling (Lanczos)")
    
    h, w = image.shape[:2]
    new_size = (w * scale, h * scale)
    
    result = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    
    logger.info(f"Fallback upscaled: {image.shape} -> {result.shape}")
    return result


def preserve_edges(
    image: np.ndarray,
    original_lineart: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """
    Enhance and preserve line art edges in the colorized image.
    
    Blends the original line art back into the colorized image
    to maintain sharp, crisp lines.
    
    Args:
        image: Colorized image (RGB).
        original_lineart: Original line art (grayscale).
        strength: Edge preservation strength (0-1).
        
    Returns:
        Image with enhanced edges.
    """
    logger.info(f"Preserving edges with strength {strength}...")
    
    # Ensure same size
    if original_lineart.shape[:2] != image.shape[:2]:
        original_lineart = cv2.resize(
            original_lineart,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # Handle grayscale lineart
    if len(original_lineart.shape) == 2:
        lineart_gray = original_lineart
    else:
        lineart_gray = cv2.cvtColor(original_lineart, cv2.COLOR_RGB2GRAY)
    
    # Invert if needed (we want dark lines to have high values for blending)
    # Detect line style by checking mean value
    if np.mean(lineart_gray) > 127:
        # White background, dark lines - invert for mask
        line_mask = 255 - lineart_gray
    else:
        # Dark background, white lines
        line_mask = lineart_gray
    
    # Convert to float for blending
    line_mask = line_mask.astype(np.float32) / 255.0
    
    # Apply strength
    line_mask = line_mask * strength
    
    # Darken image where lines are present
    result = image.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = result[:, :, c] * (1 - line_mask)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    logger.info("Edge preservation applied")
    return result


def adjust_colors(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """
    Apply color adjustments to the image.
    
    Args:
        image: Input image (RGB).
        brightness: Brightness adjustment (-1 to 1).
        contrast: Contrast multiplier (0.5 to 2.0).
        saturation: Saturation multiplier (0 to 2.0).
        
    Returns:
        Adjusted image.
    """
    logger.info(f"Adjusting colors: brightness={brightness}, "
                f"contrast={contrast}, saturation={saturation}")
    
    result = image.astype(np.float32)
    
    # Brightness
    if brightness != 0:
        result = result + brightness * 255
    
    # Contrast
    if contrast != 1.0:
        mean = np.mean(result)
        result = (result - mean) * contrast + mean
    
    # Saturation
    if saturation != 1.0:
        # Convert to HSV
        hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def postprocess_image(
    colorized: np.ndarray,
    text_layer: Optional[np.ndarray] = None,
    original_lineart: Optional[np.ndarray] = None,
    upscale: bool = False,
    upscale_factor: int = 2,
    preserve_edge_strength: float = 0.3,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    Complete post-processing pipeline.
    
    Args:
        colorized: Colorized image from colorize module.
        text_layer: Optional RGBA text layer for reinsertion.
        original_lineart: Optional line art for edge preservation.
        upscale: Whether to upscale the result.
        upscale_factor: Upscaling factor (2 or 4).
        preserve_edge_strength: Edge preservation strength (0-1).
        brightness: Brightness adjustment.
        contrast: Contrast adjustment.
        saturation: Saturation adjustment.
        use_gpu: Use GPU for upscaling.
        
    Returns:
        Final processed image (RGB).
    """
    logger.info("Starting post-processing pipeline...")
    
    result = colorized.copy()
    
    # 1. Apply color adjustments
    if brightness != 0 or contrast != 1.0 or saturation != 1.0:
        result = adjust_colors(result, brightness, contrast, saturation)
    
    # 2. Preserve edges from original line art
    if original_lineart is not None and preserve_edge_strength > 0:
        result = preserve_edges(result, original_lineart, preserve_edge_strength)
    
    # 3. Upscale if requested (before text reinsertion for better quality)
    if upscale:
        result = upscale_image(result, scale=upscale_factor, use_gpu=use_gpu)
        
        # Also upscale text layer if provided
        if text_layer is not None:
            h, w = result.shape[:2]
            text_layer = cv2.resize(
                text_layer,
                (w, h),
                interpolation=cv2.INTER_NEAREST  # Nearest for crisp text
            )
    
    # 4. Reinsert text layer (last step to ensure crisp text)
    if text_layer is not None:
        result = reinsert_text(result, text_layer)
    
    logger.info(f"Post-processing complete: shape={result.shape}")
    return result


def save_result(
    image: np.ndarray,
    output_path: str,
    quality: int = 95,
    format: str = "auto",
) -> str:
    """
    Save the final result to disk.
    
    Args:
        image: Image to save (RGB).
        output_path: Path to save to.
        quality: JPEG quality (1-100).
        format: Output format ('auto', 'png', 'jpg').
        
    Returns:
        Path to saved file.
    """
    logger.info(f"Saving result to: {output_path}")
    
    # Create directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Determine format
    if format == "auto":
        ext = Path(output_path).suffix.lower()
        if ext in [".png"]:
            format = "png"
        else:
            format = "jpg"
    
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if format == "png":
        cv2.imwrite(output_path, bgr)
    else:
        cv2.imwrite(output_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    logger.info(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        
        # Load test image
        image = cv2.imread(test_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run post-processing with fallback upscaling
        result = postprocess_image(
            image,
            upscale=True,
            upscale_factor=2,
            use_gpu=False,
        )
        
        # Save
        output_path = "./test_output/postprocessed.png"
        save_result(result, output_path)
        print(f"Saved to: {output_path}")
    else:
        print("Usage: python postprocess.py <image_path>")
