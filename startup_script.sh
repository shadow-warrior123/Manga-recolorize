#!/bin/bash
# High-Quality Manga Colorizer Setup Script for A100 VM

# 1. Update OS and Install Drivers (Only if needed)
echo "--- Installing Drivers and System Dependencies ---"
sudo apt update
sudo apt install -y nvidia-driver-535 nvidia-utils-535 libgl1-mesa-glx zip  # zip added for cleanup later

# 2. Clone the Repository
echo "--- Cloning Repository ---"
git clone https://github.com/shadow-warrior123/Manga-recolorize
cd Manga-recolorize

# 3. Create Virtual Environment
echo "--- Setting up Python Environment ---"
python3 -m venv venv
source venv/bin/activate

# 4. Install ML Stack (Force CUDA 12.1 for A100)
echo "--- Installing Torch and Dependencies (3-5 minutes) ---"
pip install --upgrade pip
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 xformers==0.0.24 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 5. Final Hardware Verification
echo "--- Verifying GPU Connection ---"
nvidia-smi
python3 -c "import torch; print('✅ GPU Visible:', torch.cuda.is_available()); print('✅ Device:', torch.cuda.get_device_name(0))"

# 6. Run Batch Processing
# NOTE: Ensure you put your manga in 'input_manga/' and references in 'reference_pages/'
echo "--- Starting Colorization Job ---"
python run_batch.py

# 7. Zip and Provide Download Link
echo "--- Preparation for Download ---"
zip -r final_results.zip output_pages/
echo "Job Complete. Use 'python3 -m http.server 8080' to download results.zip"
