"""
Multi-Reference Style Extraction Module.

Extracts color palettes, lighting characteristics, and shading information
from multiple reference images to guide the colorization process.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import cv2
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def extract_color_palette(
    image: np.ndarray,
    n_colors: int = 8,
    exclude_extremes: bool = True,
) -> List[List[int]]:
    """
    Extract a color palette from an image using K-means clustering.
    
    Args:
        image: Input image as numpy array (BGR format).
        n_colors: Number of colors to extract.
        exclude_extremes: If True, exclude near-black and near-white colors.
        
    Returns:
        List of [R, G, B] color values (0-255 range).
    """
    logger.info(f"Extracting color palette with {n_colors} colors...")
    
    # Convert to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb = image
    
    # Reshape to (N, 3) for clustering
    pixels = rgb.reshape(-1, 3).astype(np.float32)
    
    # Optionally filter out extreme values (pure black/white)
    if exclude_extremes:
        # Calculate brightness
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 20) & (brightness < 235)
        pixels = pixels[mask]
    
    if len(pixels) < n_colors:
        logger.warning("Not enough non-extreme pixels, using all pixels")
        pixels = rgb.reshape(-1, 3).astype(np.float32)
    
    # Apply K-means clustering
    kmeans = KMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(pixels)
    
    # Get cluster centers (colors)
    colors = kmeans.cluster_centers_.astype(int).tolist()
    
    # Sort by brightness (dark to light)
    colors.sort(key=lambda c: sum(c))
    
    logger.info(f"Extracted {len(colors)} colors")
    return colors


def analyze_lighting(
    image: np.ndarray,
    warm_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Analyze the lighting characteristics of an image.
    
    Determines if the image has warm (orange/yellow) or cool (blue) lighting,
    and estimates the overall brightness and contrast.
    
    Args:
        image: Input image as numpy array (BGR format).
        warm_threshold: Threshold for warm/cool classification (0-1).
        
    Returns:
        Dictionary containing:
        - 'lighting': 'warm', 'cool', or 'neutral'
        - 'brightness': Average brightness (0-255)
        - 'contrast': Contrast measure (0-1)
        - 'warm_ratio': Ratio of warm vs cool colors (0-1)
    """
    logger.info("Analyzing lighting characteristics...")
    
    # Convert to RGB and HSV for analysis
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract channels
    h, s, v = cv2.split(hsv)
    
    # Calculate brightness stats
    brightness = np.mean(v)
    contrast = np.std(v) / 128.0  # Normalized to 0-2 range
    
    # Analyze color temperature
    # Warm colors: Hue 0-60 (red-yellow) and 300-360 (magenta-red)
    # Cool colors: Hue 180-300 (cyan-blue-magenta)
    
    # Only consider saturated pixels for color temperature
    saturation_mask = s > 30
    h_saturated = h[saturation_mask]
    
    if len(h_saturated) > 0:
        # Count warm vs cool pixels
        # OpenCV hue is 0-180, so warm is 0-30 and 150-180
        warm_pixels = np.sum((h_saturated <= 30) | (h_saturated >= 150))
        cool_pixels = np.sum((h_saturated >= 90) & (h_saturated <= 150))
        total_chromatic = warm_pixels + cool_pixels
        
        if total_chromatic > 0:
            warm_ratio = warm_pixels / total_chromatic
        else:
            warm_ratio = 0.5
    else:
        warm_ratio = 0.5  # Neutral (no saturated colors)
    
    # Determine lighting classification
    if warm_ratio > warm_threshold + 0.15:
        lighting = "warm"
    elif warm_ratio < warm_threshold - 0.15:
        lighting = "cool"
    else:
        lighting = "neutral"
    
    result = {
        "lighting": lighting,
        "brightness": float(brightness),
        "contrast": float(min(contrast, 1.0)),
        "warm_ratio": float(warm_ratio),
    }
    
    logger.info(f"Lighting analysis: {lighting} (warm_ratio={warm_ratio:.2f})")
    return result


def calculate_shading_strength(
    image: np.ndarray,
) -> float:
    """
    Estimate the shading intensity/strength of an image.
    
    Analyzes the histogram distribution to determine how much
    shading/shadow detail is present in the reference.
    
    Args:
        image: Input image as numpy array (BGR format).
        
    Returns:
        Shading strength value (0.0 to 1.0).
    """
    logger.info("Calculating shading strength...")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Analyze distribution
    # Strong shading = bimodal or wide distribution
    # Flat shading = concentrated around midtones
    
    # Calculate entropy as a measure of distribution spread
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
    
    # Normalize entropy to 0-1 range (max entropy for 256 bins is 8)
    shading_strength = min(entropy / 6.0, 1.0)
    
    logger.info(f"Shading strength: {shading_strength:.2f}")
    return shading_strength


def aggregate_styles(
    reference_images: List[np.ndarray],
    palette_colors: int = 8,
) -> Dict[str, Any]:
    """
    Aggregate style information from multiple reference images.
    
    Combines palettes, averages lighting, and determines overall style.
    
    Args:
        reference_images: List of reference images (BGR format).
        palette_colors: Number of colors to extract per image.
        
    Returns:
        Aggregated style dictionary:
        {
            'palette': [[r,g,b], ...],
            'lighting': 'warm' | 'cool' | 'neutral',
            'shading_strength': float,
            'brightness': float,
            'contrast': float,
            'num_references': int,
        }
    """
    logger.info(f"Aggregating styles from {len(reference_images)} reference images...")
    
    if not reference_images:
        logger.warning("No reference images provided, using default style")
        return _get_default_style()
    
    all_colors = []
    lighting_scores = []
    shading_strengths = []
    brightnesses = []
    contrasts = []
    
    for i, image in enumerate(reference_images):
        logger.info(f"Processing reference {i + 1}/{len(reference_images)}")
        
        # Extract palette
        colors = extract_color_palette(image, n_colors=palette_colors)
        all_colors.extend(colors)
        
        # Analyze lighting
        lighting_info = analyze_lighting(image)
        lighting_scores.append(lighting_info["warm_ratio"])
        brightnesses.append(lighting_info["brightness"])
        contrasts.append(lighting_info["contrast"])
        
        # Calculate shading
        shading = calculate_shading_strength(image)
        shading_strengths.append(shading)
    
    # Aggregate colors - cluster all extracted colors
    if len(all_colors) > palette_colors:
        all_colors_array = np.array(all_colors, dtype=np.float32)
        kmeans = KMeans(n_clusters=palette_colors, random_state=42, n_init=10)
        kmeans.fit(all_colors_array)
        final_palette = kmeans.cluster_centers_.astype(int).tolist()
        final_palette.sort(key=lambda c: sum(c))
    else:
        final_palette = all_colors
    
    # Average lighting scores
    avg_warm_ratio = np.mean(lighting_scores)
    if avg_warm_ratio > 0.65:
        final_lighting = "warm"
    elif avg_warm_ratio < 0.35:
        final_lighting = "cool"
    else:
        final_lighting = "neutral"
    
    result = {
        "palette": final_palette,
        "lighting": final_lighting,
        "shading_strength": float(np.mean(shading_strengths)),
        "brightness": float(np.mean(brightnesses)),
        "contrast": float(np.mean(contrasts)),
        "warm_ratio": float(avg_warm_ratio),
        "num_references": len(reference_images),
    }
    
    logger.info(f"Style aggregation complete: {result['lighting']} lighting, "
                f"{len(result['palette'])} colors, "
                f"shading={result['shading_strength']:.2f}")
    
    return result


def _get_default_style() -> Dict[str, Any]:
    """Return a default style when no references are provided."""
    return {
        "palette": [
            [50, 50, 60],      # Dark shadow
            [100, 90, 100],    # Mid shadow
            [150, 140, 150],   # Light shadow
            [200, 190, 180],   # Light skin/highlight
            [240, 220, 200],   # Highlight 1
            [255, 200, 180],   # Warm highlight
            [180, 200, 220],   # Cool midtone
            [220, 230, 240],   # Light cool
        ],
        "lighting": "neutral",
        "shading_strength": 0.7,
        "brightness": 128.0,
        "contrast": 0.5,
        "warm_ratio": 0.5,
        "num_references": 0,
    }


def style_to_prompt_keywords(style: Dict[str, Any]) -> str:
    """
    Convert style dictionary to prompt keywords for SDXL.
    
    Generates descriptive keywords based on the extracted style.
    
    Args:
        style: Style dictionary from aggregate_styles.
        
    Returns:
        Comma-separated style keywords for the prompt.
    """
    keywords = ["high quality", "manga coloring", "anime style"]
    
    # Lighting keywords
    if style["lighting"] == "warm":
        keywords.extend(["warm lighting", "golden hour", "sunset colors"])
    elif style["lighting"] == "cool":
        keywords.extend(["cool lighting", "blue tones", "moonlight"])
    else:
        keywords.extend(["balanced lighting", "natural colors"])
    
    # Shading keywords
    if style["shading_strength"] > 0.7:
        keywords.extend(["high contrast", "dramatic shading", "cel shading"])
    elif style["shading_strength"] > 0.4:
        keywords.extend(["moderate shading", "soft shadows"])
    else:
        keywords.extend(["flat shading", "soft colors"])
    
    # Brightness keywords
    if style["brightness"] > 170:
        keywords.append("bright")
    elif style["brightness"] < 85:
        keywords.append("dark atmosphere")
    
    return ", ".join(keywords)


def load_reference_images(
    image_paths: List[str],
    max_size: int = 1024,
) -> List[np.ndarray]:
    """
    Load and preprocess reference images from file paths.
    
    Args:
        image_paths: List of paths to reference images.
        max_size: Maximum dimension for resizing (to save memory).
        
    Returns:
        List of loaded images as numpy arrays.
    """
    logger.info(f"Loading {len(image_paths)} reference images...")
    
    images = []
    for path in image_paths:
        try:
            image = cv2.imread(str(path))
            if image is None:
                logger.warning(f"Failed to load: {path}")
                continue
            
            # Resize if too large
            h, w = image.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            
            images.append(image)
            logger.info(f"Loaded: {path} ({image.shape})")
            
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
    
    return images


def process_style_extraction(
    reference_paths: List[str],
    palette_colors: int = 8,
    save_output: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete style extraction pipeline.
    
    Args:
        reference_paths: Paths to reference images.
        palette_colors: Number of colors to extract.
        save_output: Whether to save style JSON.
        output_path: Path for saved output.
        
    Returns:
        Aggregated style dictionary.
    """
    logger.info(f"Processing style extraction from {len(reference_paths)} references...")
    
    # Load images
    images = load_reference_images(reference_paths)
    
    if not images:
        logger.warning("No valid reference images loaded, using defaults")
        return _get_default_style()
    
    # Aggregate styles
    style = aggregate_styles(images, palette_colors)
    
    # Add prompt keywords
    style["prompt_keywords"] = style_to_prompt_keywords(style)
    
    # Save if requested
    if save_output and output_path:
        import json
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(style, f, indent=2)
        logger.info(f"Saved style to: {output_path}")
    
    return style


if __name__ == "__main__":
    import sys
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        style = process_style_extraction(
            paths,
            save_output=True,
            output_path="./test_output/style.json"
        )
        print(json.dumps(style, indent=2))
    else:
        print("Usage: python style_extract.py <ref1.jpg> [ref2.jpg] ...")
