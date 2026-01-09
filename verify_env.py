import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verifier")

def verify_installation():
    logger.info("Verifying installation for A100 environment...")
    
    # Check imports
    try:
        import torch
        logger.info(f"Torch version: {torch.__version__}")
        
        # Check diffusers
        import diffusers
        logger.info(f"Diffusers version: {diffusers.__version__}")
        
        # Check availability
        cuda_avail = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_avail}")
        
        if cuda_avail:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
        else:
            logger.warning("CUDA NOT DETECTED - This is expected if running on CPU-only env")
            
        # Check pipeline imports
        sys.path.append(str(Path.cwd()))
        from pipeline import device
        logger.info("Pipeline device module: OK")
        
        dtype = device.get_torch_dtype()
        logger.info(f"Recommended Dtype: {dtype}")
        
        # Check config
        from workers.gpu_worker import GPUWorker
        worker = GPUWorker()
        config = worker.config
        sdxl_config = config.get("sdxl", {})
        
        logger.info(f"Config Loaded. Model ID: {sdxl_config.get('model_id')}")
        logger.info(f"Config Loaded. SDXL Dtype: {sdxl_config.get('torch_dtype')}")
        logger.info(f"xFormers Enabled: {sdxl_config.get('enable_xformers')}")
        
        refine_config = config.get("refinement", {})
        logger.info(f"Refinement Enabled: {refine_config.get('enable')}")
        logger.info(f"Refinement Strength: {refine_config.get('strength')}")
        logger.info("VERIFICATION SUCCESSFUL: Codebase appears ready for A100.")
        return True
        
    except ImportError as e:
        logger.error(f"Import Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Verification Error: {e}")
        return False

if __name__ == "__main__":
    if verify_installation():
        sys.exit(0)
    else:
        sys.exit(1)
