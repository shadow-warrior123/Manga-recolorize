"""
Device Management Utility.

Centralizes GPU device selection, dtype management (BF16/FP16),
and memory cleanup operations for A100 optimization.
"""

import logging
import gc
import torch

logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Get the optimal device for inference.
    
    Returns:
        torch.device: 'cuda' or 'cpu'.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_torch_dtype(device: torch.device = None) -> torch.dtype:
    """
    Get the optimal floating point type for the hardware.
    
    Prefer BF16 on Ampere+ (A100), otherwise FP16.
    
    Args:
        device: Device to check capabilities for.
        
    Returns:
        torch.dtype: torch.bfloat16, torch.float16, or torch.float32.
    """
    if device is None:
        device = get_device()
        
    if device.type == "cpu":
        return torch.float32
        
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        logger.info("BF16 supported and enabled (A100 optimization)")
        return torch.bfloat16
        
    logger.info("BF16 not supported, using FP16")
    return torch.float16

def free_memory():
    """
    Aggressively free GPU memory and garbage collect.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_memory_stats() -> dict:
    """
    Get current GPU memory statistics.
    
    Returns:
        dict: Memory stats in MB.
    """
    if not torch.cuda.is_available():
        return {"type": "cpu"}
        
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }
