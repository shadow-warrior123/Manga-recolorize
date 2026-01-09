
import os
import logging
import json
import time
from pathlib import Path
from workers.gpu_worker import GPUWorker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("batch_runner")

def run_batch():
    # 1. Setup Paths
    base_dir = Path(__file__).parent.absolute()
    input_dir = base_dir / "input_manga"
    ref_dir = base_dir / "reference_pages"
    output_dir = base_dir / "output_pages"
    
    output_dir.mkdir(exist_ok=True)

    # 2. Collect Files
    valid_exts = (".jpg", ".jpeg", ".png", ".webp")
    
    pages = [str(f) for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]
    refs = [str(f) for f in ref_dir.iterdir() if f.suffix.lower() in valid_exts]

    if not pages:
        logger.error(f"No manga pages found in {input_dir}")
        return
    if not refs:
        logger.error(f"No reference images found in {ref_dir}")
        return

    logger.info(f"Found {len(pages)} pages and {len(refs)} references.")

    # 3. Initialize Worker
    # Note: This will load models into VRAM (downloads if missing)
    worker = GPUWorker(storage_root=str(base_dir / "storage"))
    
    # 4. Process Job
    job_id = f"batch_{int(time.time())}"
    logger.info(f"Starting Batch Job: {job_id}")
    
    try:
        result = worker.process_job(
            job_id=job_id,
            page_paths=pages,
            reference_paths=refs,
            options={
                "upscale": True, # High-res enabled
                "dilation_size": 15
            }
        )
        
        # 5. Move results to output_pages for easy access
        logger.info("Moving results to output_pages...")
        job_out = Path(worker.outputs_dir) / job_id
        for f in job_out.glob("*.png"):
            dest = output_dir / f.name
            os.replace(f, dest)
            logger.info(f"Saved: {dest}")

        logger.info("Batch Processing Complete!")
        
    finally:
        worker.cleanup()

if __name__ == "__main__":
    run_batch()
