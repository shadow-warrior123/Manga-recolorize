# Manga Colorizer

AI-powered manga colorization platform using SDXL + ControlNet. Transforms black & white manga pages into vibrant, anime-style colored artwork while preserving English text pixel-perfectly.

## Features

- **Text Preservation**: Automatically detects and preserves English text
- **Multi-Reference Styling**: Extract color palettes from multiple reference images
- **SDXL + ControlNet**: High-quality colorization with line art conditioning
- **Real-ESRGAN Upscaling**: Optional 2x/4x upscaling with anime model
- **REST API**: Full-featured API for job management
- **Cloud GPU Ready**: Optimized for A100 / RTX PRO 6000

## Requirements

### GPU Requirements

| Level | GPU | VRAM | Notes |
|-------|-----|------|-------|
| **Recommended** | NVIDIA A100 | 40GB+ | Full quality, fast inference |
| **Good** | RTX 6000 Ada | 48GB | Full quality |
| **Minimum** | RTX 3090/4090 | 24GB | May need reduced resolution |
| **Fallback** | CPU | N/A | Limited quality, slow |

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 50GB+ disk space (for models)

## Installation

### 1. Clone and Setup

```bash
# Navigate to project
cd manga-colorizer

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Running the API Server

### Development Mode

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Note**: Use only 1 worker for GPU inference to avoid VRAM conflicts.

### Access Points

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **GPU Info**: http://localhost:8000/gpu-info

## API Usage

### Workflow

1. **Create Job** → 2. **Upload Pages** → 3. **Upload References** → 4. **Run** → 5. **Get Results**

### Example: Using cURL

```bash
# 1. Create a new job
JOB_ID=$(curl -s -X POST http://localhost:8000/job/create | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 2. Upload manga pages
curl -X POST "http://localhost:8000/job/$JOB_ID/upload_pages" \
  -F "files=@page1.jpg" \
  -F "files=@page2.jpg"

# 3. Upload reference images (colored manga/anime)
curl -X POST "http://localhost:8000/job/$JOB_ID/upload_references" \
  -F "files=@reference1.png" \
  -F "files=@reference2.png"

# 4. Start processing
curl -X POST "http://localhost:8000/job/$JOB_ID/run"

# 5. Check status (poll until completed)
curl "http://localhost:8000/job/$JOB_ID/status"

# 6. Get results
curl "http://localhost:8000/job/$JOB_ID/results"

# 7. Download output
curl -O "http://localhost:8000/job/$JOB_ID/download/page_001_colorized.png"
```

### Example: Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# Create job
response = requests.post(f"{BASE_URL}/job/create")
job_id = response.json()["job_id"]

# Upload pages
with open("manga_page.jpg", "rb") as f:
    requests.post(
        f"{BASE_URL}/job/{job_id}/upload_pages",
        files={"files": f}
    )

# Upload references
with open("reference.png", "rb") as f:
    requests.post(
        f"{BASE_URL}/job/{job_id}/upload_references",
        files={"files": f}
    )

# Run job
requests.post(f"{BASE_URL}/job/{job_id}/run")

# Poll for completion
import time
while True:
    status = requests.get(f"{BASE_URL}/job/{job_id}/status").json()
    print(f"Progress: {status['progress_percent']}%")
    if status["status"] in ["completed", "partial", "failed"]:
        break
    time.sleep(5)

# Get results
results = requests.get(f"{BASE_URL}/job/{job_id}/results").json()
print(results)
```

## Processing Options

Pass options when creating or running a job:

```json
{
  "options": {
    "seed": 42,
    "num_steps": 30,
    "guidance_scale": 7.5,
    "upscale": true,
    "upscale_factor": 2,
    "lineart_method": "adaptive",
    "edge_strength": 0.3
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `num_steps` | 30 | Inference steps (more = better quality, slower) |
| `guidance_scale` | 7.5 | CFG scale (higher = more prompt adherence) |
| `upscale` | false | Enable Real-ESRGAN upscaling |
| `upscale_factor` | 2 | Upscale factor (2 or 4) |
| `lineart_method` | "adaptive" | Line extraction: "canny", "threshold", "adaptive", "xdog" |
| `edge_strength` | 0.3 | Line preservation strength (0-1) |

## Project Structure

```
manga-colorizer/
├── api/                    # REST API layer
│   ├── main.py            # FastAPI application
│   ├── routes.py          # API endpoints
│   └── schemas.py         # Pydantic models
├── pipeline/               # Processing modules
│   ├── text_mask.py       # Text detection & masking
│   ├── lineart.py         # Line art extraction
│   ├── style_extract.py   # Color palette extraction
│   ├── colorize.py        # SDXL + ControlNet inference
│   └── postprocess.py     # Text reinsertion & upscaling
├── workers/               # Background processing
│   └── gpu_worker.py      # GPU job orchestration
├── storage/               # File storage
│   ├── inputs/            # Uploaded manga pages
│   ├── references/        # Reference images
│   └── outputs/           # Colorized results
├── configs/
│   └── model.yaml         # Model configuration
├── requirements.txt
└── README.md
```

## Pipeline Details

### 1. Text Detection (`text_mask.py`)

- Uses CRAFT for text region detection
- Creates binary mask of text areas
- Extracts original text as RGBA layer
- Falls back to contour detection if CRAFT unavailable

### 2. Line Art Extraction (`lineart.py`)

- Multiple methods: Canny, threshold, adaptive, XDoG
- Removes text regions via inpainting
- Prepares clean line art for ControlNet

### 3. Style Extraction (`style_extract.py`)

- K-means clustering for color palette
- Analyzes warm/cool lighting
- Calculates shading intensity
- Aggregates multiple references

### 4. Colorization (`colorize.py`)

- Loads SDXL + ControlNet (lineart)
- Builds prompts from style analysis
- FP16 inference with memory optimizations
- Deterministic output with fixed seed

### 5. Post-Processing (`postprocess.py`)

- Reinserts original text pixels
- Optional Real-ESRGAN upscaling
- Edge preservation for crisp lines
- Color adjustments

## Known Limitations

1. **GPU Memory**: SDXL requires significant VRAM. Use attention slicing and VAE tiling for lower-memory GPUs.

2. **Text Detection**: Works best with clear English text. May miss stylized or handwritten text.

3. **Style Matching**: Results depend heavily on reference quality. Use similar-style references for best results.

4. **Processing Time**: ~20-60 seconds per page on A100, depending on resolution.

5. **No Training**: This is inference-only. Cannot learn new styles.

## Troubleshooting

### Out of Memory

```bash
# Reduce image size in config
# Or use CPU offloading (slower but uses less VRAM)
```

### Models Not Loading

```bash
# Check internet connection for Hugging Face downloads
# Or pre-download models:
python -c "from diffusers import StableDiffusionXLControlNetPipeline; ..."
```

### Poor Colorization Quality

- Use more/better reference images
- Increase `num_steps` (e.g., 50)
- Try different `lineart_method`
- Adjust `guidance_scale`

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Stability AI](https://stability.ai/) - SDXL model
- [lllyasviel](https://github.com/lllyasviel) - ControlNet
- [xinntao](https://github.com/xinntao) - Real-ESRGAN
- [clovaai](https://github.com/clovaai) - CRAFT text detection
