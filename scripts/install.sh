#!/usr/bin/env bash
#
# scripts/install.sh — setup environment for Ethereal Canvas

set -e

echo "=== System Prerequisite Check ==="

# Python
PYTHON=python3
$PYTHON --version

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA GPU detected"
else
    echo "No CUDA detected, continuing with CPU (may be slow)"
fi

# Create and activate virtual environment
echo "--- Creating venv ---"
$PYTHON -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu$(nvcc --version | grep -o '[0-9]\+.[0-9]\+' 2>/dev/null || echo "118")
pip install transformers gradio pillow gitpython

echo "=== Model Download ==="
python - << 'PYCODE'
from transformers import AutoConfig, AutoTokenizer
print("Downloading Qwen-Image-2512…")
AutoConfig.from_pretrained("Qwen/Qwen-Image-2512")
AutoTokenizer.from_pretrained("Qwen/Qwen-Image-2512")
print("Done")
PYCODE

echo "=== Installation Complete ==="

echo "=== Create Pinokio manifest ==="
cat <<EOF > pinokio.json
{
  "name": "ethereal_canvas",
  "version": "0.1.0",
  "entrypoint": "scripts/run.sh",
  "dependencies": ["python>=3.10"]
}
EOF