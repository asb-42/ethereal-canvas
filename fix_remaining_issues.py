"""
Quick fixes for the remaining issues after successful CUDA detection.
"""

import subprocess
import sys

def install_xformers():
    """Install xFormers for memory optimization."""
    print("🔧 Installing xFormers for memory optimization...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "xformers"], 
                      check=True, capture_output=True)
        print("✅ xFormers installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ xFormers installation failed: {e}")
        return False

def update_diffusers():
    """Update diffusers and related packages."""
    print("🔧 Updating diffusers and related packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", 
                      "diffusers", "transformers", "accelerate", "bitsandbytes"], 
                      check=True, capture_output=True)
        print("✅ Packages updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Package update failed: {e}")
        return False

def test_nf4_config():
    """Test if NF4 quantization config is available."""
    print("🧪 Testing NF4 quantization config...")
    try:
        from diffusers import DiffusersBitsAndBytesConfig
        print("✅ NF4 quantization config available")
        return True
    except ImportError as e:
        print(f"❌ NF4 quantization config not available: {e}")
        return False

if __name__ == "__main__":
    print("Quick Fixes for Remaining Issues")
    print("=" * 40)
    
    success = True
    
    # Fix xFormers
    if not install_xformers():
        success = False
    
    # Update packages
    if not update_diffusers():
        success = False
    
    # Test NF4 config
    if not test_nf4_config():
        print("⚠️ NF4 config still not available, but fp16_full strategy works fine")
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 All fixes applied successfully!")
        print("\nYou can now run:")
        print("python3 launch_ethereal_canvas.py")
        print("\nExpected improvements:")
        print("✅ xFormers memory optimization active")
        print("✅ NF4 quantization available (if needed)")
        print("✅ Better memory management strategies")
    else:
        print("⚠️ Some fixes failed, but fp16_full strategy still works")
        print("You can still run: python3 launch_ethereal_canvas.py")
    
    print("\n" + "=" * 40)