#!/usr/bin/env bash
set -euo pipefail

echo "---------------------------------------------"
echo "Ethereal Canvas Pinokio Installer"
echo "---------------------------------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$SCRIPT_DIR/.."
cd "$APP_ROOT"
echo "App root: $APP_ROOT"

# --------------------------
# Step 1: Check Python >=3.10
# --------------------------
PYTHON=$(command -v python3 || true)
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python3 >=3.10 required"
    exit 1
fi

PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
    echo "ERROR: Python >=3.10 required, found $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION detected."

# --------------------------
# Step 2: Create & activate venv
# --------------------------
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "Virtual environment activated."

# --------------------------
# Step 3: Install dependencies
# --------------------------
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

# Ensure diffusers and qwen_image packages are installed
pip install --upgrade diffusers
pip install --upgrade qwen_image || true  # optional fallback

# --------------------------
# Step 4: Download & cache models
# --------------------------
MODEL_CACHE_DIR="$APP_ROOT/models"
mkdir -p "$MODEL_CACHE_DIR"

# Function to download a model
download_model() {
    local model_name="$1"
    local folder_name="$2"
    local cache_path="$MODEL_CACHE_DIR/$folder_name"

    if [[ -d "$cache_path" ]]; then
        echo "Model $model_name already cached at $cache_path"
        return
    fi

    echo "Downloading model $model_name..."
    python - <<PYTHON
from pathlib import Path
import torch
model_dir = Path("$cache_path")
model_dir.mkdir(parents=True, exist_ok=True)

try:
    if "Edit" in "$model_name":
        from diffusers import QwenImageEditPlusPipeline
        QwenImageEditPlusPipeline.from_pretrained("$model_name", cache_dir=str(model_dir), torch_dtype=torch.float16)
    else:
        from diffusers import StableDiffusionPipeline  # placeholder for T2I
        StableDiffusionPipeline.from_pretrained("$model_name", cache_dir=str(model_dir), torch_dtype=torch.float16)
    print(f"Model {model_name} downloaded successfully.")
except Exception as e:
    print(f"ERROR downloading {model_name}: {e}")
    exit(1)
PYTHON
}

download_model "Qwen/Qwen-Image-2512" "Qwen-Image-2512"
download_model "Qwen/Qwen-Image-Edit-2511" "Qwen-Image-Edit-2511"

# --------------------------
# Step 5: Completion
# --------------------------
echo "---------------------------------------------"
echo "Installation complete!"
echo "Models cached at $MODEL_CACHE_DIR"
echo "You can now run the application:"
echo "  source .venv/bin/activate"
echo "  python run_ethereal_canvas.py"
echo "---------------------------------------------"