#!/usr/bin/env python3
"""
Debug script to test model loading without UI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """Test model loading directly."""
    print("🧪 Testing Model Loading Debug...")
    print("=" * 50)
    
    try:
        from modules.backends.text_to_image import TextToImageBackend
        
        print("✅ Import successful")
        
        # Initialize backend
        backend = TextToImageBackend()
        print(f"✅ Backend initialized: {backend.model_name}")
        print(f"   Device: {backend.device}")
        print(f"   Loaded: {backend.loaded}")
        
        # Test loading with progress
        print("\n🔄 Starting model loading...")
        backend.load()
        
        print(f"✅ Loading completed: {backend.loaded}")
        
        # Test generation
        print("\n🎨 Testing generation...")
        result = backend.generate("A simple test image")
        print(f"✅ Generation result: {result}")
        
        # Test model info
        print("\n📊 Model info:")
        info = backend.get_model_info()
        for key, value in info.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_imports():
    """Test if all required imports work."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
    except Exception as e:
        print(f"❌ torch: {e}")
        return False
    
    try:
        import diffusers
        print(f"✅ diffusers: {diffusers.__version__}")
    except Exception as e:
        print(f"❌ diffusers: {e}")
        return False
        
    try:
        from diffusers import DiffusionPipeline
        print("✅ DiffusionPipeline import successful")
    except Exception as e:
        print(f"❌ DiffusionPipeline: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except Exception as e:
        print(f"❌ transformers: {e}")
        return False
    
    return True

def main():
    """Run debug tests."""
    print("🚀 Ethereal Canvas Model Loading Debug")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed")
        return
    
    print("\n" + "=" * 50)
    
    # Test model loading
    if test_model_loading():
        print("\n✅ All tests passed!")
        print("\n🎯 If you see clean progress bars without corruption,")
        print("   the 'Getötetg checkpoint shards' issue is resolved.")
    else:
        print("\n❌ Model loading test failed")

if __name__ == "__main__":
    main()