"""
Quick test to verify memory management integration in backends.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_memory_management_integration():
    """Test if memory management is properly integrated."""
    print("=== Testing Memory Management Integration ===")
    
    # Test memory manager import
    try:
        from modules.memory import memory_manager, LoadStrategy
        print("✅ Memory manager imported successfully")
        print(f"  Available strategies: {[s.value for s in memory_manager.fallback_chain]}")
        print(f"  Device: {memory_manager.device}")
        print(f"  Available VRAM: {memory_manager.get_available_memory_mb():.1f} MB")
    except Exception as e:
        print(f"❌ Memory manager import failed: {e}")
        return False
    
    # Test backend imports
    try:
        from modules.backends.text_to_image import TextToImageBackend
        print("✅ TextToImageBackend imported successfully")
        
        # Check if memory management is available in backend
        backend = TextToImageBackend()
        if hasattr(backend, '_load_with_memory_management'):
            print("✅ Memory management methods available in TextToImageBackend")
        else:
            print("❌ Memory management methods not found in TextToImageBackend")
            return False
    except Exception as e:
        print(f"❌ TextToImageBackend import failed: {e}")
        return False
    
    try:
        from modules.backends.image_edit import ImageEditBackend
        print("✅ ImageEditBackend imported successfully")
        
        backend = ImageEditBackend()
        if hasattr(backend, '_load_with_memory_management'):
            print("✅ Memory management methods available in ImageEditBackend")
        else:
            print("❌ Memory management methods not found in ImageEditBackend")
            return False
    except Exception as e:
        print(f"❌ ImageEditBackend import failed: {e}")
        return False
    
    try:
        from modules.backends.image_inpaint import ImageInpaintBackend
        print("✅ ImageInpaintBackend imported successfully")
        
        backend = ImageInpaintBackend()
        if hasattr(backend, '_load_with_memory_management'):
            print("✅ Memory management methods available in ImageInpaintBackend")
        else:
            print("❌ Memory management methods not found in ImageInpaintBackend")
            return False
    except Exception as e:
        print(f"❌ ImageInpaintBackend import failed: {e}")
        return False
    
    print("\n=== Integration Test Summary ===")
    print("✅ All components successfully integrated with memory management")
    print("✅ Ready for testing with actual model loading")
    
    return True

if __name__ == "__main__":
    success = test_memory_management_integration()
    
    if success:
        print("\n🎉 Memory management integration test PASSED")
        print("You can now run 'python3 launch_ethereal_canvas.py' to test with real models")
    else:
        print("\n❌ Memory management integration test FAILED")
        print("Please check the errors above and fix integration issues")