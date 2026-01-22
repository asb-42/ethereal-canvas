#!/usr/bin/env python3
"""
Model downloader utility for Ethereal Canvas.
Downloads and verifies Qwen models from HuggingFace.
"""

import os
import sys
import time
from pathlib import Path
import hashlib
from datetime import datetime

def log_message(message: str):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def verify_model_directory(model_dir: Path) -> bool:
    """Verify that a model directory is properly downloaded."""
    if not model_dir.exists():
        return False
    
    # Check for essential files
    essential_files = [
        "config.json",
        "pytorch_model.bin",  # or model.safetensors
        "tokenizer.json",
        "tokenizer_config.json"
    ]
    
    for file_name in essential_files:
        file_path = model_dir / file_name
        alt_path = model_dir / "model.safetensors" if file_name == "pytorch_model.bin" else None
        
        if not file_path.exists() and (alt_path is None or not alt_path.exists()):
            log_message(f"Missing essential file: {file_name}")
            return False
    
    # Check for incomplete files
    incomplete_files = list(model_dir.rglob("*.incomplete"))
    if incomplete_files:
        log_message(f"Found {len(incomplete_files)} incomplete files")
        return False
    
    log_message(f"✅ Model directory verified: {model_dir}")
    return True

def download_qwen_model(model_name: str, cache_dir: Path) -> bool:
    """Download a Qwen model using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
        import transformers
        
        log_message(f"Starting download of {model_name}...")
        log_message(f"Cache directory: {cache_dir}")
        
        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download model with progress
        start_time = time.time()
        
        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False
        )
        
        download_time = time.time() - start_time
        log_message(f"Download completed in {download_time:.2f} seconds")
        
        # Verify download
        if verify_model_directory(Path(downloaded_path)):
            log_message(f"✅ Successfully downloaded and verified {model_name}")
            return True
        else:
            log_message(f"❌ Verification failed for {model_name}")
            return False
            
    except ImportError:
        log_message("❌ huggingface_hub or transformers not installed")
        log_message("Please install: pip install huggingface_hub transformers")
        return False
    except Exception as e:
        log_message(f"❌ Failed to download {model_name}: {e}")
        return False

def main():
    """Main function to download required models."""
    log_message("🚀 Starting model download process...")
    
    # Set up paths
    app_root = Path(__file__).parent.parent
    models_dir = app_root / "runtime" / "models"
    
    # Models to download
    models = [
        "Qwen/Qwen-Image-2512",  # Text-to-image model
        "Qwen/Qwen-Image-Edit-2511"  # Image editing model
    ]
    
    success_count = 0
    
    for model_name in models:
        model_cache_dir = models_dir / model_name.split("/")[-1]
        
        # Check if model already exists and is verified
        if verify_model_directory(model_cache_dir):
            log_message(f"✅ {model_name} already exists and verified")
            success_count += 1
            continue
        
        # Download the model
        if download_qwen_model(model_name, model_cache_dir):
            success_count += 1
        
        log_message("-" * 50)
    
    # Summary
    log_message(f"📊 Download complete: {success_count}/{len(models)} models ready")
    
    if success_count == len(models):
        log_message("🎉 All models are ready for use!")
        return True
    else:
        log_message("⚠️  Some models failed to download. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)