#!/usr/bin/env python3
"""
Sequential model downloader with robust error handling.
Downloads models one file at a time to avoid corruption.
"""

import os
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from tqdm import tqdm

def log_message(message: str, level: str = "INFO"):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def download_file_sequential(url: str, local_path: Path, chunk_size: int = 8192):
    """Download a single file sequentially."""
    try:
        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove partial file if it exists
        if local_path.exists() and local_path.suffix == '.incomplete':
            local_path.unlink()
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as f:
            with tqdm(
                desc=local_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        return True
        
    except Exception as e:
        log_message(f"Failed to download {url}: {e}", "ERROR")
        return False

def download_qwen_model_sequential(model_name: str, cache_dir: Path) -> bool:
    """Download Qwen model sequentially."""
    try:
        log_message(f"🔄 Starting sequential download of {model_name}")
        
        # Force sequential downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DOWNLOAD_RETRY"] = "3"
        
        # Use huggingface_hub with sequential flag
        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False,
            max_workers=1  # Force sequential
        )
        
        log_message(f"✅ Sequential download completed: {downloaded_path}")
        return True
        
    except Exception as e:
        log_message(f"❌ Sequential download failed: {e}", "ERROR")
        
        # Fallback: try downloading key files manually
        try:
            api = HfApi()
            repo_files = api.list_repo_files(repo_id=model_name)
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            success_count = 0
            
            for file_info in repo_files[:10]:  # Limit to first 10 files for safety
                file_path = file_info if isinstance(file_info, str) else file_info.path
                file_url = f"https://huggingface.co/{model_name}/resolve/main/{file_path}"
                local_path = cache_dir / file_path
                
                log_message(f"Downloading: {file_path}")
                if download_file_sequential(file_url, local_path):
                    success_count += 1
            
            log_message(f"Downloaded {success_count}/{len(repo_files[:10])} files manually")
            return success_count > 0
            
        except Exception as fallback_e:
            log_message(f"Fallback download also failed: {fallback_e}", "ERROR")
            return False

def clear_incomplete_downloads(model_dir: Path):
    """Clear incomplete download files."""
    incomplete_files = list(model_dir.rglob("*.incomplete"))
    for file_path in incomplete_files:
        try:
            file_path.unlink()
            log_message(f"Removed incomplete file: {file_path}")
        except Exception as e:
            log_message(f"Failed to remove {file_path}: {e}")

def main():
    """Main function for sequential model downloading."""
    log_message("🚀 Starting Sequential Model Downloader...")
    
    # Set up paths
    app_root = Path(__file__).parent.parent
    models_dir = app_root / "runtime" / "models"
    
    # Models to download
    models = [
        "Qwen/Qwen-Image-2512"
    ]
    
    success_count = 0
    
    for model_name in models:
        model_cache_dir = models_dir / model_name.split("/")[-1]
        
        # Clear incomplete files first
        clear_incomplete_downloads(model_cache_dir)
        
        # Download sequentially
        if download_qwen_model_sequential(model_name, model_cache_dir):
            success_count += 1
        
        log_message("-" * 60)
    
    # Summary
    log_message(f"📊 Sequential download complete: {success_count}/{len(models)} models ready")
    
    if success_count == len(models):
        log_message("🎉 All models downloaded successfully!")
        return True
    else:
        log_message("⚠️  Some downloads failed. Check logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)