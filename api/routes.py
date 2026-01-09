"""
API Routes Module.

REST API endpoints for manga colorization jobs.
"""

import json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from .schemas import (
    JobCreateRequest,
    JobCreateResponse,
    JobStatus,
    UploadResponse,
    RunJobRequest,
    RunJobResponse,
    JobStatusResponse,
    JobResultResponse,
    PageResult,
    StyleInfo,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

# Router instance
router = APIRouter(prefix="/job", tags=["jobs"])

# Storage paths (relative to project root)
STORAGE_ROOT = Path("storage")
INPUTS_DIR = STORAGE_ROOT / "inputs"
REFERENCES_DIR = STORAGE_ROOT / "references"
OUTPUTS_DIR = STORAGE_ROOT / "outputs"

# In-memory job state (use Redis for production)
# TODO: Replace with Redis for persistent state
_jobs: dict = {}


def _ensure_directories():
    """Ensure storage directories exist."""
    for dir_path in [INPUTS_DIR, REFERENCES_DIR, OUTPUTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


def _get_job_or_404(job_id: str) -> dict:
    """Get job data or raise 404."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return _jobs[job_id]


@router.post("/create", response_model=JobCreateResponse)
async def create_job(request: JobCreateRequest = None) -> JobCreateResponse:
    """
    Create a new colorization job.
    
    Returns a unique job ID for subsequent operations.
    """
    _ensure_directories()
    
    # Generate unique job ID
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    
    # Create job directories
    job_inputs = INPUTS_DIR / job_id
    job_refs = REFERENCES_DIR / job_id
    job_outputs = OUTPUTS_DIR / job_id
    
    job_inputs.mkdir(parents=True, exist_ok=True)
    job_refs.mkdir(parents=True, exist_ok=True)
    job_outputs.mkdir(parents=True, exist_ok=True)
    
    # Initialize job state
    now = datetime.now()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "options": request.options if request else None,
        "pages": [],
        "references": [],
        "result": None,
    }
    
    logger.info(f"Created job: {job_id}")
    
    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=now,
        message="Job created successfully. Upload pages and references, then run."
    )


@router.post("/{job_id}/upload_pages", response_model=UploadResponse)
async def upload_pages(
    job_id: str,
    files: List[UploadFile] = File(...),
) -> UploadResponse:
    """
    Upload manga page images for colorization.
    
    Accepts multiple image files (PNG, JPG, WEBP).
    """
    job = _get_job_or_404(job_id)
    
    if job["status"] != JobStatus.PENDING:
        raise HTTPException(
            status_code=400, 
            detail="Cannot upload to a job that is already running or completed"
        )
    
    job_inputs = INPUTS_DIR / job_id
    uploaded = []
    
    for file in files:
        # Validate file type
        if not file.filename:
            continue
            
        ext = Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            logger.warning(f"Skipping unsupported file: {file.filename}")
            continue
        
        # Save file
        file_path = job_inputs / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        job["pages"].append(str(file_path))
        uploaded.append(file.filename)
        logger.info(f"Uploaded page: {file.filename}")
    
    return UploadResponse(
        job_id=job_id,
        uploaded_files=uploaded,
        total_files=len(job["pages"]),
        message=f"{len(uploaded)} page(s) uploaded successfully"
    )


@router.post("/{job_id}/upload_references", response_model=UploadResponse)
async def upload_references(
    job_id: str,
    files: List[UploadFile] = File(...),
) -> UploadResponse:
    """
    Upload reference color images.
    
    Reference images are used to extract color palette and style.
    Multiple references will be aggregated.
    """
    job = _get_job_or_404(job_id)
    
    if job["status"] != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail="Cannot upload to a job that is already running or completed"
        )
    
    job_refs = REFERENCES_DIR / job_id
    uploaded = []
    
    for file in files:
        if not file.filename:
            continue
            
        ext = Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            logger.warning(f"Skipping unsupported file: {file.filename}")
            continue
        
        file_path = job_refs / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        job["references"].append(str(file_path))
        uploaded.append(file.filename)
        logger.info(f"Uploaded reference: {file.filename}")
    
    return UploadResponse(
        job_id=job_id,
        uploaded_files=uploaded,
        total_files=len(job["references"]),
        message=f"{len(uploaded)} reference(s) uploaded successfully"
    )


def _run_job_sync(job_id: str, options: dict = None):
    """
    Run job processing synchronously.
    
    This runs in a background task.
    """
    from workers.gpu_worker import GPUWorker
    
    job = _jobs.get(job_id)
    if not job:
        logger.error(f"Job not found for background processing: {job_id}")
        return
    
    logger.info(f"Starting background processing for job: {job_id}")
    
    job["status"] = JobStatus.PROCESSING
    job["started_at"] = datetime.now()
    
    try:
        worker = GPUWorker(storage_root=str(STORAGE_ROOT))
        
        # Merge options
        merged_options = job.get("options") or {}
        if options:
            merged_options.update(options)
        
        result = worker.process_job(
            job_id=job_id,
            page_paths=job["pages"],
            reference_paths=job["references"],
            options=merged_options,
        )
        
        job["result"] = result
        job["status"] = JobStatus.COMPLETED if result.get("pages_failed", 0) == 0 else JobStatus.PARTIAL
        job["completed_at"] = datetime.now()
        
        logger.info(f"Job completed: {job_id}")
        
    except Exception as e:
        logger.error(f"Job failed: {job_id} - {e}")
        job["status"] = JobStatus.FAILED
        job["completed_at"] = datetime.now()
        job["result"] = {"error": str(e)}


@router.post("/{job_id}/run", response_model=RunJobResponse)
async def run_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    request: RunJobRequest = None,
) -> RunJobResponse:
    """
    Start processing the job.
    
    Requires at least one page and one reference to be uploaded.
    Processing runs in the background.
    """
    job = _get_job_or_404(job_id)
    
    if job["status"] != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is already {job['status'].value}"
        )
    
    if not job["pages"]:
        raise HTTPException(
            status_code=400,
            detail="No pages uploaded. Upload at least one page first."
        )
    
    if not job["references"]:
        raise HTTPException(
            status_code=400,
            detail="No references uploaded. Upload at least one reference first."
        )
    
    # Start background processing
    options = request.options if request else None
    background_tasks.add_task(_run_job_sync, job_id, options)
    
    # Update status
    job["status"] = JobStatus.PROCESSING
    
    logger.info(f"Job queued for processing: {job_id}")
    
    return RunJobResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        pages_count=len(job["pages"]),
        references_count=len(job["references"]),
        message="Processing started. Check status for progress."
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Get the current status of a job.
    """
    job = _get_job_or_404(job_id)
    
    result = job.get("result", {})
    pages_total = len(job["pages"])
    pages_completed = result.get("pages_completed", 0)
    pages_failed = result.get("pages_failed", 0)
    
    progress = (pages_completed + pages_failed) / pages_total * 100 if pages_total > 0 else 0
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        pages_total=pages_total,
        pages_completed=pages_completed,
        pages_failed=pages_failed,
        progress_percent=round(progress, 1),
        elapsed_seconds=result.get("elapsed_seconds"),
        errors=result.get("errors", []),
    )


@router.get("/{job_id}/results", response_model=JobResultResponse)
async def get_job_results(job_id: str) -> JobResultResponse:
    """
    Get the results of a completed job.
    
    Returns paths to output files and style information.
    """
    job = _get_job_or_404(job_id)
    
    if job["status"] == JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail="Job has not been run yet"
        )
    
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=400,
            detail="Job is still processing"
        )
    
    result = job.get("result", {})
    
    # Build page results
    pages = []
    for output in result.get("outputs", []):
        pages.append(PageResult(
            page_num=output.get("page_num", 0),
            input_file=Path(output.get("input", "")).name,
            output_file=Path(output.get("output", "")).name if output.get("output") else None,
            status=output.get("status", "unknown"),
            error=output.get("error"),
        ))
    
    # Build style info
    style_info = None
    if "style" in result:
        style = result["style"]
        style_info = StyleInfo(
            palette=style.get("palette", []),
            lighting=style.get("lighting", "neutral"),
            shading_strength=style.get("shading_strength", 0.5),
            num_references=style.get("num_references", 0),
        )
    
    return JobResultResponse(
        job_id=job_id,
        status=job["status"],
        pages=pages,
        style=style_info,
        output_directory=str(OUTPUTS_DIR / job_id),
        elapsed_seconds=result.get("elapsed_seconds"),
    )


@router.get("/{job_id}/download/{filename}")
async def download_output(job_id: str, filename: str):
    """
    Download a specific output file.
    """
    _get_job_or_404(job_id)
    
    file_path = OUTPUTS_DIR / job_id / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="image/png"
    )


@router.delete("/{job_id}")
async def delete_job(job_id: str) -> dict:
    """
    Delete a job and all its files.
    """
    job = _get_job_or_404(job_id)
    
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a job that is still processing"
        )
    
    # Remove files
    for dir_path in [INPUTS_DIR / job_id, REFERENCES_DIR / job_id, OUTPUTS_DIR / job_id]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    # Remove from state
    del _jobs[job_id]
    
    logger.info(f"Deleted job: {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}
