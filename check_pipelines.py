#!/usr/bin/env python3
"""
Check what pipelines are available for Qwen models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_available_pipelines():
    """Check what pipelines are available in diffusers."""
    try:
        import diffusers
        print(f"🔍 Diffusers version: {diffusers.__version__}")
        
        # Check available pipelines
        from diffusers import AutoPipelineForText2Image
        print("✅ AutoPipelineForText2Image available")
        
        # Check if QwenImagePipeline exists
        try:
            from diffusers import QwenImagePipeline
            print("✅ QwenImagePipeline available")
        except ImportError:
            print("❌ QwenImagePipeline not available")
        
        # Check DiffusionPipeline
        try:
            from diffusers import DiffusionPipeline
            print("✅ DiffusionPipeline available")
        except ImportError:
            try:
                from diffusers.pipelines.pipeline_utils import DiffusionPipeline
                print("✅ DiffusionPipeline available (alternate import)")
            except ImportError:
                print("❌ DiffusionPipeline not available")
        
        # Test AutoPipeline for Qwen model
        try:
            print("\n🔄 Testing AutoPipeline for Qwen...")
            pipeline = AutoPipelineForText2Image.from_pretrained(
                "Qwen/Qwen-Image-2512",
                torch_dtype="auto"
            )
            print("✅ AutoPipeline successful!")
            
            # Check pipeline type
            print(f"📊 Pipeline type: {type(pipeline)}")
            print(f"📊 Pipeline class: {pipeline.__class__.__name__}")
            
        except Exception as e:
            print(f"❌ AutoPipeline failed: {e}")
        
        # Test direct DiffusionPipeline
        try:
            print("\n🔄 Testing DiffusionPipeline for Qwen...")
            from diffusers import DiffusionPipeline
            pipeline = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image-2512",
                torch_dtype="auto"
            )
            print("✅ DiffusionPipeline successful!")
            
            # Check pipeline type
            print(f"📊 Pipeline type: {type(pipeline)}")
            print(f"📊 Pipeline class: {pipeline.__class__.__name__}")
            
        except Exception as e:
            print(f"❌ DiffusionPipeline failed: {e}")
            
    except ImportError as e:
        print(f"❌ Diffusers not available: {e}")

def main():
    """Check pipeline availability."""
    print("🚀 Pipeline Availability Check for Qwen Models")
    print("=" * 60)
    
    check_available_pipelines()
    
    print("\n💡 Recommendations:")
    print("  - Use AutoPipelineForText2Image for automatic detection")
    print("  - Or check if QwenImagePipeline is available")
    print("  - Avoid deprecated torch_dtype parameter")

if __name__ == "__main__":
    main()