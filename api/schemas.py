"""
API Schemas Module.

Pydantic models for request/response validation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class JobCreateRequest(BaseModel):
    """Request schema for creating a new job."""
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional processing options"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "options": {
                    "upscale": True,
                    "upscale_factor": 2,
                    "seed": 42,
                }
            }
        }


class JobCreateResponse(BaseModel):
    """Response schema for job creation."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Creation timestamp")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "pending",
                "created_at": "2024-01-01T12:00:00Z",
                "message": "Job created successfully"
            }
        }


class UploadResponse(BaseModel):
    """Response schema for file uploads."""
    job_id: str = Field(..., description="Job identifier")
    uploaded_files: List[str] = Field(..., description="List of uploaded file names")
    total_files: int = Field(..., description="Total files for this type")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "uploaded_files": ["page1.jpg", "page2.jpg"],
                "total_files": 2,
                "message": "2 pages uploaded successfully"
            }
        }


class RunJobRequest(BaseModel):
    """Request schema for running a job."""
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Override processing options"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "options": {
                    "num_steps": 30,
                    "guidance_scale": 7.5,
                }
            }
        }


class RunJobResponse(BaseModel):
    """Response schema for job run initiation."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    pages_count: int = Field(..., description="Number of pages to process")
    references_count: int = Field(..., description="Number of reference images")
    message: str = Field(..., description="Status message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "processing",
                "pages_count": 5,
                "references_count": 3,
                "message": "Processing started"
            }
        }


class PageResult(BaseModel):
    """Result schema for a single processed page."""
    page_num: int = Field(..., description="Page number")
    input_file: str = Field(..., description="Input file name")
    output_file: Optional[str] = Field(None, description="Output file name")
    status: str = Field(..., description="Processing status")
    error: Optional[str] = Field(None, description="Error message if failed")


class JobStatusResponse(BaseModel):
    """Response schema for job status query."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    pages_total: int = Field(0, description="Total pages to process")
    pages_completed: int = Field(0, description="Pages successfully processed")
    pages_failed: int = Field(0, description="Pages that failed")
    progress_percent: float = Field(0, description="Progress percentage")
    elapsed_seconds: Optional[float] = Field(None, description="Processing time")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "processing",
                "pages_total": 10,
                "pages_completed": 5,
                "pages_failed": 0,
                "progress_percent": 50.0,
            }
        }


class StyleInfo(BaseModel):
    """Style information extracted from references."""
    palette: List[List[int]] = Field(..., description="Color palette (RGB values)")
    lighting: str = Field(..., description="Lighting type (warm/cool/neutral)")
    shading_strength: float = Field(..., description="Shading intensity (0-1)")
    num_references: int = Field(..., description="Number of references used")


class JobResultResponse(BaseModel):
    """Response schema for job results."""
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Final job status")
    pages: List[PageResult] = Field(default_factory=list, description="Page results")
    style: Optional[StyleInfo] = Field(None, description="Extracted style info")
    output_directory: str = Field(..., description="Output directory path")
    elapsed_seconds: Optional[float] = Field(None, description="Total processing time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "completed",
                "pages": [
                    {
                        "page_num": 1,
                        "input_file": "page1.jpg",
                        "output_file": "page_001_colorized.png",
                        "status": "success"
                    }
                ],
                "output_directory": "storage/outputs/job_abc123",
                "elapsed_seconds": 45.2,
            }
        }


class ErrorResponse(BaseModel):
    """Generic error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    job_id: Optional[str] = Field(None, description="Related job ID if applicable")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "NotFound",
                "message": "Job not found",
                "job_id": "job_abc123"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="GPU availability")
    models_loaded: bool = Field(..., description="Model load status")
