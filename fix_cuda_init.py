"""
CUDA initialization fix for PyTorch 2.10.0+cu128 compatibility issues.
This script patches the CUDA initialization problem.
"""

import os
import sys

def apply_cuda_fix():
    """Apply CUDA initialization fix for PyTorch compatibility."""
    print("🔧 Applying CUDA initialization fix...")
    
    # Set environment variables to fix CUDA initialization
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    # Suppress specific CUDA warnings that are confusing
    import warnings
    warnings.filterwarnings("ignore", message=".*CUDA initialization.*forward compatibility.*")
    warnings.filterwarnings("ignore", message=".*UserWarning: CUDA is not available.*")
    
    print("✅ CUDA fix applied")

def force_cuda_detection():
    """Force CUDA detection even with initialization warnings."""
    print("🔧 Forcing CUDA detection...")
    
    try:
        import torch
        
        # Force CUDA initialization
        if hasattr(torch.cuda, '_lazy_init'):
            torch.cuda._lazy_init()
        
        # Check again after forced init
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        print(f"✅ CUDA available after fix: {cuda_available}")
        print(f"✅ CUDA device count: {device_count}")
        
        if cuda_available and device_count > 0:
            props = torch.cuda.get_device_properties(0)
            print(f"✅ GPU: {props.name} ({props.total_memory / (1024**3):.1f}GB)")
            return True
        else:
            print("❌ Still no CUDA devices after fix")
            return False
            
    except Exception as e:
        print(f"❌ CUDA fix failed: {e}")
        return False

if __name__ == "__main__":
    print("CUDA Initialization Fix")
    print("=" * 40)
    
    apply_cuda_fix()
    success = force_cuda_detection()
    
    if success:
        print("\n🎉 CUDA fix successful!")
        print("You can now run: python3 launch_ethereal_canvas.py")
    else:
        print("\n❌ CUDA fix failed")
        print("You may need to:")
        print("1. Reinstall PyTorch with correct CUDA version")
        print("2. Use CPU-only PyTorch if GPU issues persist")