"""
Colorization Module.

Core colorization using SDXL + ControlNet for manga pages.
Applies line art conditioning and style-based prompts.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from . import device as device_utils

logger = logging.getLogger(__name__)

# Global model cache
_model_cache: Dict[str, Any] = {}


def load_models(
    sdxl_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet_model_id: str = "lllyasviel/sd-controlnet-lineart",
    device: Optional[str] = None,
    torch_dtype: str = "auto",
    enable_attention_slicing: bool = True,
    enable_vae_tiling: bool = True,
    offload_to_cpu: bool = False,
    enable_xformers: bool = True,
) -> Dict[str, Any]:
    """
    Load SDXL and ControlNet models for colorization.
    
    Uses model caching to avoid reloading on subsequent calls.
    
    Args:
        sdxl_model_id: Hugging Face model ID for SDXL base.
        controlnet_model_id: Hugging Face model ID for ControlNet.
        device: Device to load models on ('cuda' or 'cpu').
        torch_dtype: Data type for inference ('float16' or 'float32').
        enable_attention_slicing: Enable attention slicing for memory savings.
        enable_vae_tiling: Enable VAE tiling for large images.
        offload_to_cpu: Offload model parts to CPU when not in use.
        
    Returns:
        Dictionary containing loaded models and pipeline.
    """
    global _model_cache
    
    cache_key = f"{sdxl_model_id}_{controlnet_model_id}"
    
    if cache_key in _model_cache:
        logger.info("Using cached models")
        return _model_cache[cache_key]
    
    logger.info(f"Loading models: SDXL={sdxl_model_id}, ControlNet={controlnet_model_id}")
    
    # Determine device and dtype
    target_device = device_utils.get_device() if device is None else torch.device(device)
    
    if torch_dtype == "auto":
        dtype = device_utils.get_torch_dtype(target_device)
    elif torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    logger.info(f"Using device: {target_device}, dtype: {dtype}")
    
    try:
        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            AutoencoderKL,
        )
        
        # Load ControlNet
        logger.info("Loading ControlNet model...")
        
        # Try SDXL-compatible ControlNet first
        try:
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-canny-sdxl-1.0",
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception as e:
            logger.warning(f"Failed to load SDXL ControlNet, trying alternative: {e}")
            # Fallback to SD 1.5 compatible ControlNet (will need adapter)
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=dtype,
            )
        
        # Load VAE for better quality
        logger.info("Loading VAE...")
        try:
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=dtype,
            )
        except Exception as e:
            logger.warning(f"Failed to load custom VAE: {e}")
            vae = None
        
        # Load SDXL pipeline with ControlNet
        logger.info("Loading SDXL pipeline...")
        pipeline_kwargs = {
            "controlnet": controlnet,
            "torch_dtype": dtype,
            "use_safetensors": True,
            "variant": "fp16" if torch_dtype == "float16" else None,
        }
        
        if vae is not None:
            pipeline_kwargs["vae"] = vae
        
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            sdxl_model_id,
            **pipeline_kwargs,
        )

        # Switch to Euler Ancestral (typical for anime models)
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # Move to device
        pipe = pipe.to(target_device)
        
        # Enable xFormers if requested and available
        if enable_xformers:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers memory efficient attention")
            except Exception as e:
                logger.warning(f"Could not enable xFormers: {e}")
        
        # Enable memory optimizations
        if enable_attention_slicing:
            pipe.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        
        if enable_vae_tiling:
            pipe.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
        
        if offload_to_cpu:
            pipe.enable_model_cpu_offload()
            logger.info("Enabled CPU offload")
        
        # Create Refinement Pipeline (Img2Img)
        # Shares the same components, so zero extra VRAM
        from diffusers import StableDiffusionXLImg2ImgPipeline
        
        refine_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(pipe)
        
        models = {
            "pipeline": pipe,
            "refine_pipeline": refine_pipe,
            "controlnet": controlnet,
            "device": target_device,
            "dtype": dtype,
        }
        
        _model_cache[cache_key] = models
        logger.info("Models loaded successfully")
        
        return models
        
    except ImportError as e:
        logger.error(f"Required libraries not installed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise


def colorize_page(
    lineart: np.ndarray,
    text_mask: np.ndarray,
    style: Dict[str, Any],
    models: Dict[str, Any],
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 0.8,
    negative_prompt: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None,
    # Refinement args
    # Refinement args
    enable_refinement: bool = False,
    refinement_strength: float = 0.35,
    refinement_scale: float = 1.5,
    dry_run: bool = False,
) -> np.ndarray:
    """
    Colorize a manga page using SDXL + ControlNet.
    
    Args:
        lineart: Line art image as numpy array (RGB).
        text_mask: Binary mask of text regions.
        style: Style dictionary from style extraction.
        models: Loaded models from load_models().
        seed: Random seed for reproducibility.
        num_inference_steps: Number of denoising steps.
        guidance_scale: CFG scale for prompt guidance.
        controlnet_conditioning_scale: ControlNet influence strength.
        negative_prompt: Negative prompt for undesired features.
        target_size: Optional (width, height) for output.
        
    Returns:
        Colorized image as numpy array (RGB).
    """
    logger.info("Starting colorization...")
    
    pipe = models["pipeline"]
    device = models["device"]
    dtype = models["dtype"]
    
    # Prepare control image (line art)
    if len(lineart.shape) == 2:
        lineart_rgb = np.stack([lineart] * 3, axis=-1)
    else:
        lineart_rgb = lineart
    
    # Convert to PIL Image
    control_image = Image.fromarray(lineart_rgb.astype(np.uint8))
    
    # Determine output size
    if target_size is None:
        # Round to nearest multiple of 8
        w, h = control_image.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        target_size = (w, h)
    
    control_image = control_image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Build prompt from style
    prompt = _build_colorization_prompt(style)
    logger.info(f"Using prompt: {prompt[:100]}...")
    
    # Default negative prompt
    # Enhanced negative prompt for anime
    if negative_prompt is None:
        negative_prompt = (
            "blurry, low quality, distorted, watermark, signature, "
            "bad anatomy, deformed, ugly, text, caption, jpeg artifacts, "
            "(monochrome:1.3), (greyscale:1.3), (realistic:1.2), photo, real life, "
            "sketch, pencil, bad hands, missing fingers"
        )
    
    # Set generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Run inference
    # Run inference
    logger.info(f"Running inference: {num_inference_steps} steps, size={target_size}")
    
    if dry_run:
        logger.warning("DRY RUN: Skipping actual inference, generating random noise.")
        # Create random noise image (RGB)
        colorized = np.random.randint(0, 255, (target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Determine upscale path (logic check)
        if enable_refinement:
            logger.info(f"DRY RUN Refinement: scale={refinement_scale}, strength={refinement_strength}")
        
        # Apply text mask logic and return
        colorized = _apply_text_mask(colorized, lineart_rgb, text_mask)
        return colorized
    
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )
        
        output_image = result.images[0]
        colorized = np.array(output_image)
        
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU Out of Memory! Trying to clear cache and retry...")
        device_utils.free_memory()
        raise  # Re-raise to let worker handle fallback
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise
    
    # --- Refinement Pass (High-Res Fix) ---
    if enable_refinement:
        logger.info(f"Running refinement pass: scale={refinement_scale}, strength={refinement_strength}")
        try:
            refine_pipe = models.get("refine_pipeline")
            if refine_pipe:
                # 1. Upscale initial result
                import cv2
                h, w = colorized.shape[:2]
                new_size = (int(w * refinement_scale), int(h * refinement_scale))
                
                # Use Lanczos for clean input
                upscaled_input = cv2.resize(colorized, new_size, interpolation=cv2.INTER_LANCZOS4)
                upscaled_input_pil = Image.fromarray(upscaled_input)
                
                # 2. Run Img2Img
                refine_result = refine_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=upscaled_input_pil,
                    num_inference_steps=int(num_inference_steps * 0.8), # Fewer steps needed
                    strength=refinement_strength,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                
                colorized = np.array(refine_result.images[0])
                logger.info(f"Refinement complete: {colorized.shape}")
                
                # Resize lineart/mask for final assembly if needed?
                # Actually, postprocess handles resizing, but _apply_text_mask needs matching sizes
                # We need to resize lineart and text_mask to match the new colorized size
                # for the step below
                target_h, target_w = colorized.shape[:2]
                
                lineart_rgb = cv2.resize(lineart_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
                if text_mask is not None:
                    text_mask = cv2.resize(text_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    
            else:
                logger.warning("Refinement pipeline not available, skipping")
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            # Continue with original result
    
    # Apply text mask if needed
    colorized = _apply_text_mask(colorized, lineart_rgb, text_mask)
    
    logger.info(f"Colorization complete: shape={colorized.shape}")
    return colorized


def _build_colorization_prompt(style: Dict[str, Any]) -> str:
    """
    Build a prompt for colorization based on extracted style.
    
    Args:
        style: Style dictionary from style extraction.
        
    Returns:
        Prompt string for SDXL.
    """
    # Base prompt components with quality boosters
    base = [
        "masterpiece",
        "best quality",
        "very aesthetic",
        "absurdres",
        "highly detailed manga coloring",
        "anime illustration style",
        "professional digital coloring",
        "vibrant colors",
        "cel shading",
    ]
    
    # Add style keywords if present
    if "prompt_keywords" in style:
        base.append(style["prompt_keywords"])
    
    # Add lighting-specific terms
    lighting = style.get("lighting", "neutral")
    if lighting == "warm":
        base.extend(["warm color palette", "golden lighting", "sunset atmosphere"])
    elif lighting == "cool":
        base.extend(["cool color palette", "blue lighting", "moonlit atmosphere"])
    else:
        base.extend(["balanced lighting", "natural colors"])
    
    # Add shading terms
    shading = style.get("shading_strength", 0.5)
    if shading > 0.7:
        base.append("dramatic shading and shadows")
    elif shading > 0.4:
        base.append("soft cel shading")
    else:
        base.append("flat anime coloring")
    
    return ", ".join(base)


def _apply_text_mask(
    colorized: np.ndarray,
    lineart: np.ndarray,
    text_mask: np.ndarray,
) -> np.ndarray:
    """
    Ensure text regions preserve original content.
    
    During colorization, text areas should remain uncolored
    to allow for clean text reinsertion later.
    
    Args:
        colorized: Colorized image.
        lineart: Original line art.
        text_mask: Binary mask of text regions.
        
    Returns:
        Colorized image with text regions handled.
    """
    if text_mask is None or np.sum(text_mask) == 0:
        return colorized
    
    # Resize mask if needed
    if text_mask.shape[:2] != colorized.shape[:2]:
        import cv2
        text_mask = cv2.resize(
            text_mask,
            (colorized.shape[1], colorized.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    
    # Create white background in text areas for clean reinsertion
    result = colorized.copy()
    mask_bool = text_mask > 127
    
    # Fill text areas with white
    if len(result.shape) == 3:
        for c in range(result.shape[2]):
            result[:, :, c][mask_bool] = 255
    else:
        result[mask_bool] = 255
    
    return result


def colorize_with_fallback(
    lineart: np.ndarray,
    text_mask: np.ndarray,
    style: Dict[str, Any],
    **kwargs,
) -> np.ndarray:
    """
    Colorize with automatic model loading and error handling.
    
    This is the main entry point for colorization, handling
    model loading and providing a simple colorization fallback
    if GPU inference fails.
    
    Args:
        lineart: Line art image.
        text_mask: Binary text mask.
        style: Style dictionary.
        **kwargs: Additional arguments for colorize_page.
        
    Returns:
        Colorized image.
    """
    try:
        models = load_models()
        return colorize_page(lineart, text_mask, style, models, **kwargs)
    except Exception as e:
        logger.error(f"GPU colorization failed: {e}")
        logger.info("Using fallback colorization")
        return _fallback_colorize(lineart, style)


def _fallback_colorize(
    lineart: np.ndarray,
    style: Dict[str, Any],
) -> np.ndarray:
    """
    Simple fallback colorization using palette application.
    
    This is a stub for when GPU inference is unavailable.
    It applies a basic color overlay based on the style palette.
    
    # TODO: Implement more sophisticated CPU-based colorization
    
    Args:
        lineart: Line art image (grayscale or RGB).
        style: Style dictionary with palette.
        
    Returns:
        Basic colorized image.
    """
    import cv2
    
    logger.warning("Using fallback colorization (limited quality)")
    
    # Convert to grayscale if needed
    if len(lineart.shape) == 3:
        gray = cv2.cvtColor(lineart, cv2.COLOR_RGB2GRAY)
    else:
        gray = lineart
    
    # Get base color from palette (use a mid-tone)
    palette = style.get("palette", [[200, 180, 160]])
    base_color = np.array(palette[len(palette) // 2]) if palette else np.array([200, 180, 160])
    
    # Create a colored version using luminosity
    h, w = gray.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply base color modulated by luminosity
    for i, c in enumerate(base_color):
        colored[:, :, i] = (gray.astype(float) / 255 * c).astype(np.uint8)
    
    # Keep dark lines as lines
    line_mask = gray < 50
    for i in range(3):
        colored[:, :, i][line_mask] = gray[line_mask]
    
    return colored


def unload_models():
    """
    Unload models from memory and clear cache.
    
    Call this to free GPU memory after processing is complete.
    """
    global _model_cache
    
    logger.info("Unloading models...")
    
    for key, models in _model_cache.items():
        if "pipeline" in models:
            del models["pipeline"]
        if "controlnet" in models:
            del models["controlnet"]
    
    _model_cache.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Models unloaded")


if __name__ == "__main__":
    import sys
    import cv2
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
        
        # Load test image
        lineart = cv2.imread(test_image)
        lineart = cv2.cvtColor(lineart, cv2.COLOR_BGR2RGB)
        
        # Create empty mask (no text)
        text_mask = np.zeros(lineart.shape[:2], dtype=np.uint8)
        
        # Default style
        style = {
            "palette": [[100, 80, 60], [200, 180, 160], [240, 220, 200]],
            "lighting": "neutral",
            "shading_strength": 0.6,
        }
        
        # Try colorization
        try:
            result = colorize_with_fallback(lineart, text_mask, style)
            
            # Save result
            output_path = "./test_output/colorized.png"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"Saved result to: {output_path}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python colorize.py <lineart_image>")
