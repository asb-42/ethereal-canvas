"""
Aggressive CUDA fix for PyTorch 2.10.0+cu128 compatibility.
This addresses the persistent CUDA initialization error.
"""

import os
import sys
import subprocess

def apply_aggressive_cuda_fix():
    """Apply aggressive CUDA fix for PyTorch 2.10.0+cu128."""
    print("🔧 Applying AGGRESSIVE CUDA fix for PyTorch 2.10.0+cu128...")
    
    # Set environment variables BEFORE importing anything
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Force specific CUDA version if available
    cuda_version = os.environ.get("CUDA_VERSION", "")
    if not cuda_version:
        try:
            # Try to detect CUDA version
            result = subprocess.run(["nvcc", "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Detected CUDA: {result.stdout.strip()}")
        except:
            print("⚠️ Could not detect CUDA version")
    
    print("✅ Aggressive CUDA environment variables set")
    return True

def test_cuda_after_fix():
    """Test CUDA detection after applying fixes."""
    print("🧪 Testing CUDA after aggressive fix...")
    
    # Apply fixes
    apply_aggressive_cuda_fix()
    
    try:
        # Import torch after fixes are applied
        import torch
        import warnings
        
        # Filter warnings more aggressively
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
        warnings.filterwarnings("ignore", message=".*forward compatibility.*")
        warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
        
        print(f"✅ PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✅ CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA device count: {device_count}")
            
            if device_count > 0:
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / (1024**3)
                print(f"✅ GPU: {props.name} ({memory_gb:.1f}GB)")
                print(f"✅ Compute capability: {props.major}.{props.minor}")
                
                # Test actual CUDA operation
                try:
                    test_tensor = torch.randn(10, 10).cuda()
                    print("✅ CUDA operation test successful!")
                    del test_tensor
                    torch.cuda.empty_cache()
                    return True
                except Exception as e:
                    print(f"❌ CUDA operation test failed: {e}")
                    return False
            else:
                print("❌ No CUDA devices found")
                return False
        else:
            print("❌ CUDA not available after fixes")
            
            # Check if we can force it
            try:
                print("🔧 Attempting to force CUDA initialization...")
                torch.cuda.init()
                device_count = torch.cuda.device_count()
                print(f"✅ Forced CUDA device count: {device_count}")
                return device_count > 0
            except Exception as e:
                print(f"❌ Force CUDA failed: {e}")
                return False
    
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def suggest_alternatives():
    """Suggest alternatives if CUDA still doesn't work."""
    print("\n" + "="*60)
    print("ALTERNATIVE SOLUTIONS")
    print("="*60)
    
    print("\n1. Reinstall PyTorch with CUDA 11.8 compatibility:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n2. Use CPU-only PyTorch (slower but stable):")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    
    print("\n3. Update CUDA drivers:")
    print("   Ensure you have CUDA 11.8 or 12.x installed")
    print("   Check with: nvidia-smi")
    
    print("\n4. Use the CPU fallback (memory management will still work):")
    print("   CPU fallback is optimized and should work reasonably well")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    print("AGGRESSIVE CUDA FIX FOR PYTORCH 2.10.0+CU128")
    print("="*60)
    
    success = test_cuda_after_fix()
    
    if success:
        print("\n🎉 AGGRESSIVE CUDA FIX SUCCESSFUL!")
        print("You can now run: python3 launch_ethereal_canvas.py")
        print("\nThe system should:")
        print("✅ Detect your 23.6GB GPU")
        print("✅ Use CUDA for acceleration") 
        print("✅ Apply memory management with GPU strategies")
        print("✅ Load models successfully with automatic optimization")
    else:
        print("\n❌ AGGRESSIVE CUDA FIX FAILED")
        suggest_alternatives()
    
    print("\n" + "="*60)