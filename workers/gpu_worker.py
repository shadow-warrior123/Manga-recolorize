"""
GPU Worker Module.

Orchestrates the full colorization pipeline for processing jobs.
Designed to run on cloud GPU instances (A100, RTX PRO 6000).
"""

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import numpy as np
import cv2

# Import pipeline modules
from pipeline import text_mask, lineart, style_extract, colorize, postprocess, device

logger = logging.getLogger(__name__)


class GPUWorker:
    """
    GPU worker for processing manga colorization jobs.
    
    Handles the full pipeline from input images to final colorized output.
    """
    
    def __init__(
        self,
        storage_root: str = "./storage",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the GPU worker.
        
        Args:
            storage_root: Root directory for storage.
            config_path: Optional path to model.yaml config.
        """
        self.storage_root = Path(storage_root)
        self.inputs_dir = self.storage_root / "inputs"
        self.references_dir = self.storage_root / "references"
        self.outputs_dir = self.storage_root / "outputs"
        self.temp_dir = self.storage_root / "temp"
        
        # Create directories
        for dir_path in [self.inputs_dir, self.references_dir, 
                         self.outputs_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Models will be loaded on first use
        self.models = None
        self.models_loaded = False
        
        logger.info(f"GPU Worker initialized. Storage: {storage_root}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "model.yaml"
        
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from: {config_path}")
            return config
        else:
            logger.warning(f"Config not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "sdxl": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "torch_dtype": "auto",
            },
            "controlnet": {
                "model_id": "lllyasviel/sd-controlnet-lineart",
                "conditioning_scale": 0.8,
            },
            "inference": {
                "seed": 42,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "image_size": 1024,
            },
            "style": {
                "palette_colors": 8,
            },
            "upscaler": {
                "scale": 2,
            },
        }
    
    def load_models(self) -> None:
        """
        Pre-load models for faster processing.
        
        Call this before processing to avoid loading during job execution.
        """
        if self.models_loaded:
            logger.info("Models already loaded")
            return
        
        logger.info("Loading models...")
        
        try:
            self.models = colorize.load_models(
                sdxl_model_id=self.config.get("sdxl", {}).get(
                    "model_id", "stabilityai/stable-diffusion-xl-base-1.0"
                ),
                controlnet_model_id=self.config.get("controlnet", {}).get(
                    "model_id", "lllyasviel/sd-controlnet-lineart"
                ),
                torch_dtype=self.config.get("sdxl", {}).get("torch_dtype", "auto"),
                enable_attention_slicing=True,
                enable_vae_tiling=True,
                enable_xformers=self.config.get("sdxl", {}).get("enable_xformers", True),
            )
            self.models_loaded = True
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models = None
            self.models_loaded = False
    
    def process_job(
        self,
        job_id: str,
        page_paths: List[str],
        reference_paths: List[str],
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a complete colorization job.
        
        Args:
            job_id: Unique job identifier.
            page_paths: List of paths to manga page images.
            reference_paths: List of paths to reference color images.
            options: Optional processing options.
            
        Returns:
            Job result dictionary with status and output paths.
        """
        logger.info(f"Starting job: {job_id}")
        logger.info(f"Pages: {len(page_paths)}, References: {len(reference_paths)}")
        
        start_time = time.time()
        options = options or {}
        
        # Create job output directory
        job_output_dir = self.outputs_dir / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        result = {
            "job_id": job_id,
            "status": "processing",
            "started_at": datetime.now().isoformat(),
            "pages_total": len(page_paths),
            "pages_completed": 0,
            "pages_failed": 0,
            "outputs": [],
            "errors": [],
        }
        
        # Step 1: Extract style from references
        logger.info("Extracting style from reference images...")
        try:
            style = style_extract.process_style_extraction(
                reference_paths,
                palette_colors=self.config.get("style", {}).get("palette_colors", 8),
                save_output=True,
                output_path=str(job_output_dir / "style.json"),
            )
            result["style"] = style
        except Exception as e:
            logger.error(f"Style extraction failed: {e}")
            style = style_extract._get_default_style()
            result["style"] = style
            result["errors"].append(f"Style extraction failed, using defaults: {e}")
        
        # Load models if not already loaded
        if not self.models_loaded:
            self.load_models()
        
        # Step 2: Process each page
        for i, page_path in enumerate(page_paths):
            page_num = i + 1
            logger.info(f"Processing page {page_num}/{len(page_paths)}: {page_path}")
            
            try:
                output_path = job_output_dir / f"page_{page_num:03d}_colorized.png"
                
                page_result = self.process_single_page(
                    page_path=page_path,
                    style=style,
                    output_path=str(output_path),
                    options=options,
                )
                
                result["outputs"].append({
                    "page_num": page_num,
                    "input": page_path,
                    "output": str(output_path),
                    "status": "success",
                })
                result["pages_completed"] += 1
                
            except Exception as e:
                import torch
                if isinstance(e, torch.cuda.OutOfMemoryError):
                    logger.error(f"OOM Error on page {page_num}! Attempting to clear memory.")
                    device.free_memory()
                
                error_msg = f"Page {page_num} failed: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                
                result["errors"].append(error_msg)
                result["outputs"].append({
                    "page_num": page_num,
                    "input": page_path,
                    "output": None,
                    "status": "failed",
                    "error": str(e),
                })
                result["pages_failed"] += 1
        
        # Finalize result
        elapsed_time = time.time() - start_time
        result["status"] = "completed" if result["pages_failed"] == 0 else "partial"
        result["completed_at"] = datetime.now().isoformat()
        result["elapsed_seconds"] = round(elapsed_time, 2)
        
        # Save result JSON
        result_path = job_output_dir / "result.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Job {job_id} completed: {result['pages_completed']}/{result['pages_total']} pages")
        logger.info(f"Elapsed time: {elapsed_time:.2f}s")
        
        return result
    
    def process_single_page(
        self,
        page_path: str,
        style: Dict[str, Any],
        output_path: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single manga page through the full pipeline.
        
        Args:
            page_path: Path to the manga page image.
            style: Style dictionary from style extraction.
            output_path: Path to save the output.
            options: Processing options.
            
        Returns:
            Processing result for this page.
        """
        options = options or {}
        
        # Load image
        image = cv2.imread(page_path)
        if image is None:
            raise ValueError(f"Failed to load image: {page_path}")
        
        logger.info(f"Loaded image: {image.shape}")
        
        # Step 1: Text detection and masking
        logger.info("Step 1: Text detection...")
        text_result = text_mask.process_text_detection(
            page_path,
            use_cuda=True,
            save_mask=options.get("save_intermediates", False),
            save_layer=True,
            output_dir=str(Path(output_path).parent / "intermediates"),
            dilation_size=options.get("dilation_size", 15),
        )
        
        text_mask_array = text_result["mask"]
        text_layer = text_result["text_layer"]
        
        # Step 2: Line art extraction
        logger.info("Step 2: Line art extraction...")
        lineart_image = lineart.process_lineart_extraction(
            page_path,
            text_mask=text_mask_array,
            method=options.get("lineart_method", "adaptive"),
            target_size=None,  # Keep original size
            save_output=options.get("save_intermediates", False),
            output_path=str(Path(output_path).parent / "intermediates" / "lineart.png"),
        )
        
        # Step 3: Colorization
        logger.info("Step 3: Colorization...")
        inference_config = self.config.get("inference", {})
        controlnet_config = self.config.get("controlnet", {})
        
        if self.models is not None:
            colorized = colorize.colorize_page(
                lineart=lineart_image,
                text_mask=text_mask_array,
                style=style,
                models=self.models,
                seed=options.get("seed", inference_config.get("seed", 42)),
                num_inference_steps=options.get(
                    "num_steps", inference_config.get("num_inference_steps", 30)
                ),
                guidance_scale=options.get(
                    "guidance_scale", inference_config.get("guidance_scale", 7.5)
                ),
                controlnet_conditioning_scale=controlnet_config.get("conditioning_scale", 0.8),
                # Refinement Config
                enable_refinement=self.config.get("refinement", {}).get("enable", False),
                refinement_strength=self.config.get("refinement", {}).get("strength", 0.35),
                refinement_scale=self.config.get("refinement", {}).get("scale_factor", 1.5),
                dry_run=options.get("dry_run", False),
            )
        else:
            # Fallback colorization
            logger.warning("Using fallback colorization (no GPU models)")
            colorized = colorize.colorize_with_fallback(
                lineart_image, text_mask_array, style
            )
        
        # Step 4: Post-processing
        logger.info("Step 4: Post-processing...")
        upscaler_config = self.config.get("upscaler", {})
        
        final_image = postprocess.postprocess_image(
            colorized=colorized,
            text_layer=text_layer,
            original_lineart=lineart_image,
            upscale=options.get("upscale", upscaler_config.get("enable", False)),
            upscale_factor=options.get("upscale_factor", upscaler_config.get("scale", 4)),
            preserve_edge_strength=options.get("edge_strength", 0.3),
            use_gpu=True,
        )
        
        # Save result
        postprocess.save_result(
            final_image,
            output_path,
            quality=options.get("quality", 95),
        )
        
        logger.info(f"Saved: {output_path}")
        
        return {
            "output_path": output_path,
            "output_shape": final_image.shape,
        }
    
    def cleanup(self) -> None:
        """
        Clean up resources and unload models.
        """
        logger.info("Cleaning up GPU worker...")
        
        if self.models_loaded:
            colorize.unload_models()
            self.models = None
            self.models_loaded = False
            
        device.free_memory()
        
        logger.info("Cleanup complete")


def process_job(
    job_id: str,
    page_paths: List[str],
    reference_paths: List[str],
    storage_root: str = "./storage",
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to process a job.
    
    Creates a worker, processes the job, and cleans up.
    
    Args:
        job_id: Unique job identifier.
        page_paths: List of manga page paths.
        reference_paths: List of reference image paths.
        storage_root: Storage root directory.
        options: Processing options.
        
    Returns:
        Job result dictionary.
    """
    worker = GPUWorker(storage_root=storage_root)
    
    try:
        result = worker.process_job(job_id, page_paths, reference_paths, options)
    finally:
        worker.cleanup()
    
    return result


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Simple CLI for testing
    if len(sys.argv) >= 3:
        pages = sys.argv[1].split(",")
        refs = sys.argv[2].split(",")
        job_id = sys.argv[3] if len(sys.argv) > 3 else f"test_{int(time.time())}"
        
        result = process_job(job_id, pages, refs)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python gpu_worker.py <page1,page2,...> <ref1,ref2,...> [job_id]")
        print("Example: python gpu_worker.py input1.jpg,input2.jpg ref1.png,ref2.png test_job")
