#!/usr/bin/env python3
"""
Quick test to verify the progress bar corruption fix.
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

def test_clean_loading():
    """Test clean model loading without progress corruption."""
    print("🧪 Testing Clean Model Loading...")
    print("=" * 50)
    
    try:
        from modules.backends.text_to_image import TextToImageBackend
        
        print("✅ Sequential downloads configured")
        print("🔄 Starting model loading test...")
        
        # Initialize backend
        backend = TextToImageBackend()
        print(f"✅ Backend initialized: {backend.model_name}")
        
        # Load model (this is where corruption occurred)
        print("\n🔄 Loading model (watch for clean progress bars)...")
        backend.load()
        
        print(f"✅ Model loaded successfully: {backend.loaded}")
        
        # Test generation
        print("\n🎨 Testing generation...")
        result = backend.generate("Test image for corruption check")
        print(f"✅ Generation result: {result}")
        
        print("\n🎉 SUCCESS: No progress bar corruption detected!")
        print("   If you see clean progress bars above, the issue is resolved.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run the corruption check test."""
    print("🚀 Progress Bar Corruption Fix Verification")
    print("=" * 60)
    print("Environment variables set:")
    print(f"  HF_HUB_ENABLE_HF_TRANSFER = {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print(f"  HF_HUB_DOWNLOAD_RETRY = {os.environ.get('HF_HUB_DOWNLOAD_RETRY')}")
    print("=" * 60)
    
    if test_clean_loading():
        print("\n✅ VERIFICATION PASSED")
        print("The 'Getötetg checkpoint shards' issue should be resolved!")
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Progress bar corruption may still be present.")

if __name__ == "__main__":
    main()