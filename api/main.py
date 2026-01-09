"""
Manga Colorizer API - Main Application.

FastAPI application for manga colorization service.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .routes import router as job_router
from .schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# API version
API_VERSION = "1.0.0"

# Storage setup
STORAGE_ROOT = Path("storage")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Sets up storage directories and optionally pre-loads models.
    """
    logger.info("Starting Manga Colorizer API...")
    
    # Create storage directories
    for subdir in ["inputs", "references", "outputs", "temp"]:
        (STORAGE_ROOT / subdir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Storage initialized: {STORAGE_ROOT.absolute()}")
    
    # Log GPU status
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("No GPU available. Colorization will use fallback mode.")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Manga Colorizer API...")
    
    # Unload models if loaded
    try:
        from pipeline.colorize import unload_models
        unload_models()
    except Exception:
        pass


# Create FastAPI application
app = FastAPI(
    title="Manga Colorizer API",
    description="""
    AI-powered manga colorization service using SDXL + ControlNet.
    """,
    version=API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(job_router)


@app.get("/", tags=["root"])
async def root():
    """Redirect to UI dashboard."""
    return RedirectResponse(url="/ui/index.html")


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    models_loaded = False
    try:
        from pipeline.colorize import _model_cache
        models_loaded = len(_model_cache) > 0
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        gpu_available=gpu_available,
        models_loaded=models_loaded,
    )


@app.get("/gpu-info", tags=["health"])
async def gpu_info():
    """Get detailed GPU information."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "No GPU available"}
    
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        ),
        "allocated_memory_gb": round(
            torch.cuda.memory_allocated(0) / (1024**3), 2
        ),
        "cached_memory_gb": round(
            torch.cuda.memory_reserved(0) / (1024**3), 2
        ),
    }

# Mount static files for UI and storage
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
