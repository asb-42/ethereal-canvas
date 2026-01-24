#!/bin/bash
echo "🔧 Comprehensive CUDA Environment Fix"
echo "=================================================="

# Set CUDA environment variables globally (before Python starts)
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

echo "✅ CUDA environment variables set globally"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Set environment variables in virtual environment
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# Update/Install packages with xformers
echo "🔧 Installing/updating packages with xformers support..."
pip install -U diffusers transformers accelerate
pip install xformers

# Test CUDA detection in virtual environment
echo "🧪 Testing CUDA in virtual environment..."
python3 -c "
import os
import warnings
import torch

print('🔧 CUDA Environment Variables Set:')
print(f'  CUDA_LAUNCH_BLOCKING: {os.environ.get(\"CUDA_LAUNCH_BLOCKING\", \"Not set\")}')
print(f'  PYTORCH_CUDA_ALLOC_CONF: {os.environ.get(\"PYTORCH_CUDA_ALLOC_CONF\", \"Not set\")}')

warnings.filterwarnings('ignore', message='.*CUDA initialization.*forward compatibility.*')
warnings.filterwarnings('ignore', message='.*UserWarning: CUDA is not available.*')
warnings.filterwarnings('ignore', message='.*torch_dtype.*is deprecated.*')

try:
    cuda_available = torch.cuda.is_available()
    print(f'✅ CUDA available: {cuda_available}')
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f'✅ CUDA device count: {device_count}')
        
        if device_count > 0:
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            print(f'✅ GPU: {props.name} ({memory_gb:.1f}GB)')
            
            # Test NF4 config
            try:
                from diffusers import DiffusersBitsAndBytesConfig
                print('✅ DiffusersBitsAndBytesConfig available')
                nf4_config = DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=['transformer_blocks.0.img_mod'],
                )
                print('✅ NF4 quantization config created successfully')
            except Exception as e:
                print(f'⚠️  NF4 config issue: {e}')
        
        print('✅ CUDA functionality test passed!')
    else:
        print('❌ CUDA not available')
        
except Exception as e:
    print(f'❌ Error during CUDA test: {e}')
"

echo "=================================================="
echo "✅ Environment setup complete!"

# Now launch the main application
echo "🚀 Starting Ethereal Canvas with optimized environment..."
echo "=================================================="

python3 launch_ethereal_canvas.py