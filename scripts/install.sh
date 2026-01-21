#!/usr/bin/env bash
set -euo pipefail

echo "---------------------------------------------"
echo "Ethereal Canvas Developer Installer"
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

# --------------------------
# Step 4: Developer Options
# --------------------------
echo "Developer options:"
echo "1. Download models now (recommended for testing)"
echo "2. Skip models (download on first use)"
echo -n "Choose option [1-2]: "
read -r choice

if [[ "$choice" == "1" ]]; then
    echo "Downloading models..."
    MODEL_CACHE_DIR="$APP_ROOT/models"
    mkdir -p "$MODEL_CACHE_DIR"
    
    # Note: Model downloads will be handled by the backends on first use
    echo "Models will be downloaded to $MODEL_CACHE_DIR on first use."
else
    echo "Skipping model downloads. Models will be downloaded on first use."
fi

# --------------------------
# Step 5: Completion
# --------------------------
echo "---------------------------------------------"
echo "Developer installation complete!"
echo "Virtual environment: .venv"
echo "Activate with: source .venv/bin/activate"
echo "Run application: python run_ethereal_canvas.py"
echo "---------------------------------------------"