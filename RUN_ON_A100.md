# Running on A100 (NVIDIA Ampere)

This codebase has been optimized for NVIDIA A100 GPUs. It utilizes **BF16 (BFloat16)** precision, **xFormers** memory-efficient attention, and **Animagine XL 3.1** for premium anime quality.

## Prerequisites

- **OS**: Ubuntu 22.04 (Recommended)
- **Driver**: NVIDIA Driver 535+
- **CUDA**: 12.1+

## 1. Environment Setup

It is highly recommended to use a virtual environment or Conda environment.

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (Pinned for A100/CUDA 12.x)
pip install -r requirements.txt
```

## 2. Configuration

The configuration is located at `configs/model.yaml`. Key A100 settings:

```yaml
sdxl:
  model_id: "cagliostrolab/animagine-xl-3.1"  # Premium Anime Model
  torch_dtype: "auto"       # Automatically selects "bfloat16" on A100
  enable_xformers: true     # Use memory-efficient attention

inference:
  guidance_scale: 8.0       # Higher guidance for strong anime style

upscaler:
  enable: true              # 4x Upscaling enabled by default
```

## 3. Verifying GPU Access

Run the following python snippet to verify your environment is correctly detecting the A100 and enabling optimizations:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
```

## 4. Running Inference

To run a job using the GPU worker:

```bash
# Syntax: python workers/gpu_worker.py <pages> <refs> <job_id>
python workers/gpu_worker.py page1.jpg,page2.jpg ref1.jpg job_001
```

## 5. Troubleshooting

### Out of Memory (OOM)
If you encounter OOM errors despite optimizations:
1. **Reduce Image Size**: Set `inference.image_size` to 1024 (default) or lower in `configs/model.yaml`.
2. **Disable VAE Tiling**: In rare cases, tiling adds overhead. Set `enable_vae_tiling: false`.
3. **Use FP16**: If BF16 uses slightly more memory (uncommon), force `torch_dtype: "float16"`.

### "xFormers not available"
If you see warnings about xFormers:
- Ensure it is installed: `pip install xformers`
- Ensure it matches your torch version (see `requirements.txt`).
- It is optional; the code will run without it (slower).
