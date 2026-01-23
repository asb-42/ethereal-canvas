#!/bin/bash
echo "🔧 Quick fixes for remaining issues (run in venv)"
echo "=================================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install xFormers
echo "🔧 Installing xFormers..."
pip install xformers

# Update packages
echo "🔧 Updating diffusers and related packages..."
pip install -U diffusers transformers accelerate bitsandbytes

# Test NF4 config
echo "🧪 Testing NF4 quantization config..."
python -c "
try:
    from diffusers import DiffusersBitsAndBytesConfig
    print('✅ NF4 quantization config available')
except ImportError as e:
    print(f'❌ NF4 quantization config not available: {e}')
"

echo "=================================================="
echo "🎉 Fixes applied! Now run:"
echo "source venv/bin/activate"
echo "python3 launch_ethereal_canvas.py"