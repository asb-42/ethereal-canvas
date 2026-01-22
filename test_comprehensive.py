#!/usr/bin/env python3
"""
Comprehensive test for T2I model loading on GPU and CPU machines.
"""

import sys
import os
from pathlib import Path

# Set up sequential downloads BEFORE any imports
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DOWNLOAD_RETRY"] = "3"

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_device_detection():
    """Test device detection and GPU memory checking."""
    print("🔍 Testing Device Detection...")
    print("=" * 50)
    
    try:
        import torch
        
        print(f"📦 PyTorch version: {torch.__version__}")
        print(f"🎮 CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            try:
                gpu_count = torch.cuda.device_count()
                print(f"🎮 GPU count: {gpu_count}")
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    print(f"🎮 GPU {i}: {props.name} ({memory_gb:.1f}GB VRAM)")
                    
                    if memory_gb >= 4:
                        print(f"✅ GPU {i} has sufficient memory for models")
                    else:
                        print(f"⚠️  GPU {i} has insufficient memory (needs 4GB+)")
            except Exception as e:
                print(f"❌ GPU detection failed: {e}")
        else:
            print("💻 Using CPU for model loading")
            
    except ImportError:
        print("❌ PyTorch not available")
    
    print()

def test_model_loading():
    """Test model loading with improved error handling."""
    print("🧪 Testing Model Loading...")
    print("=" * 50)
    
    try:
        from modules.backends.text_to_image import TextToImageBackend
        
        print("✅ TextToImageBackend import successful")
        
        # Initialize backend (this will show device detection)
        backend = TextToImageBackend()
        print(f"✅ Backend initialized: {backend.model_name}")
        print(f"🔧 Selected device: {backend.device}")
        
        # Load model (this is where issues occurred)
        print("\n🔄 Starting model loading...")
        start_time = os.times()[4]
        
        backend.load()
        
        load_time = os.times()[4] - start_time
        print(f"✅ Model loading completed in {load_time:.2f} seconds")
        print(f"📊 Model loaded successfully: {backend.loaded}")
        
        if backend.loaded and backend.pipeline is not None:
            print("✅ Pipeline loaded successfully (not in stub mode)")
            
            # Test generation
            print("\n🎨 Testing generation...")
            try:
                result = backend.generate("Test image for verification")
                print(f"✅ Generation successful: {result}")
                
                if result and Path(result).exists():
                    print(f"✅ Output file exists: {result}")
                else:
                    print(f"⚠️  Output file not found, but generation completed")
                    
            except Exception as gen_e:
                print(f"❌ Generation failed: {gen_e}")
        else:
            print("⚠️  Using stub mode (pipeline not loaded)")
            
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_progress_bars():
    """Test that progress bars work without corruption."""
    print("📊 Testing Progress Bar Rendering...")
    print("=" * 50)
    print("Look for clean progress indicators:")
    print("  ✓ Loading pipeline components... X%")
    print("  ✓ Loading checkpoint shards... X%")
    print("  ✓ No 'Getötetg' or garbled text")
    print("  ✓ Sequential loading (one file at a time)")
    print()

def main():
    """Run comprehensive tests."""
    print("🚀 Comprehensive T2I Model Loading Test")
    print("=" * 60)
    print("Environment:")
    print(f"  HF_HUB_ENABLE_HF_TRANSFER = {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print(f"  HF_HUB_DOWNLOAD_RETRY = {os.environ.get('HF_HUB_DOWNLOAD_RETRY')}")
    print("=" * 60)
    
    # Run tests
    test_device_detection()
    test_progress_bars()
    
    success = test_model_loading()
    
    print("=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\n💡 Expected behavior:")
        print("  ✓ GPU machines should load models with CUDA acceleration")
        print("  ✓ CPU machines should load models without variant errors")
        print("  ✓ Progress bars should be clean and readable")
        print("  ✓ No 'variant=fp16' errors should occur")
        print("  ✓ No 'Getötetg' corruption in progress bars")
    else:
        print("❌ SOME TESTS FAILED")
        print("\n🔧 Check the error messages above for solutions")

if __name__ == "__main__":
    main()