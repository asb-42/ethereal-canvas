#!/usr/bin/env bash
set -euo pipefail

# -------------------------------------------------------
# Ethereal Canvas - Pinokio One-Click Installer Script
# -------------------------------------------------------

echo "---------------------------------------------"
echo "Welcome to Ethereal Canvas Pinokio Installer"
echo "---------------------------------------------"

# Determine script and app root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$SCRIPT_DIR/.."
cd "$APP_ROOT"
echo "App root directory: $APP_ROOT"

# -------------------------------------------------------
# Step 1: Detect Python >=3.10
# -------------------------------------------------------
PYTHON=$(command -v python3 || true)
if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python3 not found. Pinokio requires Python >=3.10."
    exit 1
fi

PYTHON_VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if (( $(echo "$PYTHON_VERSION < 3.10" | bc -l) )); then
    echo "ERROR: Python >=3.10 required, found $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION detected."

# -------------------------------------------------------
# Step 2: Create & activate virtual environment
# -------------------------------------------------------
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# -------------------------------------------------------
# Step 3: Install dependencies
# -------------------------------------------------------
if [[ ! -f "requirements.txt" ]]; then
    echo "ERROR: requirements.txt not found in $APP_ROOT"
    exit 1
fi

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

# -------------------------------------------------------
# Step 4: Download and cache Qwen-Image model
# -------------------------------------------------------
MODEL_NAME="Qwen/Qwen-Image-2512"
MODEL_CACHE_DIR="$APP_ROOT/models"
mkdir -p "$MODEL_CACHE_DIR"

python - <<PYTHON
from pathlib import Path

model_dir = Path("$MODEL_CACHE_DIR") / "Qwen-Image-2512"

if not model_dir.exists():
    print(f"Downloading model {MODEL_NAME} to {model_dir}")
    try:
        # Try Qwen-specific loader if installed
        from qwen_image import QwenImageForConditionalGeneration
        model = QwenImageForConditionalGeneration.from_pretrained(
            "$MODEL_NAME", cache_dir=str(model_dir)
        )
    except ImportError:
        # Fall back to generic AutoModel loader
        print("qwen_image package not installed; falling back to AutoModelForImageGeneration")
        from transformers import AutoModelForImageGeneration, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("$MODEL_NAME", cache_dir=str(model_dir))
        model = AutoModelForImageGeneration.from_pretrained("$MODEL_NAME", cache_dir=str(model_dir))
else:
    print(f"Model already cached at {model_dir}")
PYTHON

# -------------------------------------------------------
# Step 5: Final message
# -------------------------------------------------------
echo "---------------------------------------------"
echo "Ethereal Canvas installation via Pinokio complete!"
echo "Python version: $($PYTHON --version)"
echo "Installed packages:"
pip list
echo "Qwen-Image-2512 model cached at: $MODEL_CACHE_DIR"
echo ""
echo "To launch the app from Pinokio:"
echo "  source .venv/bin/activate"
echo "  python run_ethereal_canvas.py"
echo "---------------------------------------------"