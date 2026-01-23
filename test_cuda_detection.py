"""
Quick test to verify CUDA detection and memory management are working correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cuda_detection():
    """Test CUDA detection and device setup."""
    print("=== Testing CUDA Detection ===")
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✅ CUDA device count: {device_count}")
            
            if device_count > 0:
                props = torch.cuda.get_device_properties(0)
                print(f"✅ GPU: {props.name}")
                print(f"✅ Memory: {props.total_memory / (1024**3):.1f} GB")
                print(f"✅ Compute Capability: {props.major}.{props.minor}")
                
                # Test simple CUDA operation
                try:
                    x = torch.randn(3, 3).cuda()
                    print("✅ CUDA operation successful")
                    del x
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"❌ CUDA operation failed: {e}")
                    return False
            else:
                print("❌ No CUDA devices found")
                return False
        else:
            print("❌ CUDA not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    return True

def test_memory_manager():
    """Test memory manager functionality."""
    print("\n=== Testing Memory Manager ===")
    
    try:
        from modules.memory import memory_manager, LoadStrategy
        print(f"✅ Memory manager device: {memory_manager.device}")
        print(f"✅ Available strategies: {[s.value for s in memory_manager.fallback_chain]}")
        
        # Test strategy recommendation
        available_memory = memory_manager.get_available_memory_mb()
        print(f"✅ Available memory: {available_memory:.1f} MB")
        
        recommendations = memory_manager.get_recommended_strategy(available_memory)
        print(f"✅ Recommended strategies: {[s.value for s in recommendations]}")
        
        return True
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False

def test_backend_device_detection():
    """Test backend device detection."""
    print("\n=== Testing Backend Device Detection ===")
    
    try:
        from modules.backends.text_to_image import TextToImageBackend
        backend = TextToImageBackend()
        print(f"✅ Backend device: {backend.device}")
        print(f"✅ Backend initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Backend device detection failed: {e}")
        return False

if __name__ == "__main__":
    print("CUDA Detection and Memory Management Test")
    print("=" * 50)
    
    results = []
    results.append(("CUDA Detection", test_cuda_detection()))
    results.append(("Memory Manager", test_memory_manager()))
    results.append(("Backend Detection", test_backend_device_detection()))
    
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou should now be able to run:")
        print("python3 launch_ethereal_canvas.py")
        print("\nThe memory management system should automatically:")
        print("• Detect and use your GPU")
        print("• Select appropriate memory strategies")
        print("• Load models successfully on your 23.6GB GPU")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above before running the main application.")
    
    print("\n" + "=" * 50)