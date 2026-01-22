"""
Runtime path utilities for Ethereal Canvas.

Provides centralized path management for all runtime artifacts.
Enforces Opencode runtime hygiene standards.
"""

from pathlib import Path
from datetime import datetime
import os

# -------------------------------------------------
# Canonical runtime root
# -------------------------------------------------
RUNTIME_ROOT = Path("runtime")

# -------------------------------------------------
# Runtime directories
# -------------------------------------------------
LOGS_DIR = RUNTIME_ROOT / "logs"
OUTPUTS_DIR = RUNTIME_ROOT / "outputs"
CACHE_DIR = RUNTIME_ROOT / "cache"
TMP_DIR = RUNTIME_ROOT / "tmp"

# -------------------------------------------------
# Model cache directories
# -------------------------------------------------
MODEL_CACHE_DIR = RUNTIME_ROOT / ".." / "models"
QWEN_T2I_CACHE = MODEL_CACHE_DIR / "Qwen-Image-2512"
QWEN_I2I_CACHE = MODEL_CACHE_DIR / "Qwen-Image-Edit-2511"

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def ensure_runtime_dirs() -> None:
    """Create all runtime directories if they do not exist."""
    for path in (LOGS_DIR, OUTPUTS_DIR, CACHE_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
    """Generate consistent timestamp for filenames."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def output_image_path(prefix: str, suffix: str = "png") -> Path:
    """Generate unique output image path."""
    ensure_runtime_dirs()
    return OUTPUTS_DIR / f"{prefix}_{timestamp()}.{suffix}"

def output_edit_path(prefix: str, suffix: str = "png") -> Path:
    """Generate unique edit image path."""
    ensure_runtime_dirs()
    return OUTPUTS_DIR / f"{prefix}_{timestamp()}.{suffix}"

def output_inpaint_path(prefix: str, suffix: str = "png") -> Path:
    """Generate unique inpaint image path."""
    ensure_runtime_dirs()
    return OUTPUTS_DIR / f"{prefix}_{timestamp()}.{suffix}"

def log_file_path(name: str, suffix: str = "log") -> Path:
    """Generate log file path."""
    ensure_runtime_dirs()
    return LOGS_DIR / f"{name}_{timestamp()}.{suffix}"

def tmp_path(name: str = None, suffix: str = "") -> Path:
    """Generate temporary file path."""
    ensure_runtime_dirs()
    if name and suffix:
        return TMP_DIR / f"{name}_{timestamp()}.{suffix}"
    elif name:
        return TMP_DIR / f"{name}_{timestamp()}"
    else:
        return TMP_DIR / timestamp()

def model_cache_path(model_name: str) -> Path:
    """Generate model cache path."""
    ensure_runtime_dirs()
    return MODEL_CACHE_DIR / model_name

def cleanup_temp_files() -> None:
    """Clean all temporary files older than 1 hour."""
    if not TMP_DIR.exists():
        return
    
    import time
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    try:
        for file_path in TMP_DIR.iterdir():
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                if file_mtime < one_hour_ago:
                    file_path.unlink()
                    print(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        print(f"Error cleaning temp files: {e}")

def get_cache_usage() -> dict:
    """Get cache directory usage statistics."""
    cache_stats = {}
    for cache_name in ["Qwen-Image-2512", "Qwen-Image-Edit-2511"]:
        cache_path = model_cache_path(cache_name)
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            cache_stats[cache_name] = {
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": len(list(cache_path.rglob("*")))
            }
        else:
            cache_stats[cache_name] = {"size_bytes": 0, "size_mb": 0, "file_count": 0}
    
    return cache_stats