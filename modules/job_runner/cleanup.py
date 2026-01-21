"""
Disk cleanup policy for output management.
"""

import os
import glob
from pathlib import Path

OUTPUT_DIR = Path("outputs")

def cleanup_old_outputs(max_keep=10):
    """
    Keep only the last N most recent output files.
    """
    if not OUTPUT_DIR.exists():
        return
    
    # Get all output files with modification times
    files = []
    for pattern in ["*.png", "*.jpg", "*.jpeg"]:
        files.extend(glob.glob(str(OUTPUT_DIR / pattern)))
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old files
    for old_file in files[max_keep:]:
        try:
            os.remove(old_file)
        except OSError:
            pass  # File might be locked