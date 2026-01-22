# Upload handling utilities for Ethereal Canvas

import shutil
import tempfile
from pathlib import Path
from modules.runtime.paths import tmp_path

def safe_copy_upload(src_path: Path, filename: str = None) -> Path:
    """
    Safely copy uploaded file to runtime/tmp/ directory.
    
    Args:
        src_path: Path to uploaded file
        filename: Optional filename (uses original if None)
    
    Returns:
        Path: Path to copied file in runtime/tmp/
    """
    if not filename:
        filename = src_path.name
    
    tmp_dir = Path(tmp_path())
    dest_path = tmp_dir / filename
    
    # Copy file with proper error handling
    try:
        shutil.copy2(src_path, dest_path)
        return dest_path
    except Exception as e:
        raise IOError(f"Failed to copy upload: {e}")

def cleanup_upload(path: Path) -> None:
    """Clean up uploaded file from runtime/tmp/."""
    try:
        path.unlink(missing_ok=True)
    except Exception as e:
        print(f"[uploads] Warning: Failed to cleanup upload {path}: {e}")