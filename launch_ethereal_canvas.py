#!/usr/bin/env python3
"""
Launcher script for Ethereal Canvas with proper error handling and status reporting.
"""

import os
import sys
import time
from pathlib import Path
import traceback
from datetime import datetime
import warnings

# Set CUDA environment variables BEFORE any imports
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Import warnings suppression
warnings.filterwarnings("ignore", message=".*UserWarning: CUDA is not available.*") 
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*zero_cond_t.*")
warnings.filterwarnings("ignore", message=".*config.json.*")
warnings.filterwarnings("ignore", message=".*Getötet.*")

def setup_sequential_downloads():
    """Set up environment for sequential downloads to prevent corruption."""
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Disable parallel transfer
    os.environ["HF_HUB_DOWNLOAD_RETRY"] = "3"  # Retry downloads
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry
    os.environ["HUGGINGFACE_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable conflicting progress bars
    os.environ["HF_HUB_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars completely
    log_message("✅ Configured sequential downloads")

def check_dependencies():
    """Check if required dependencies are available."""
    missing_deps = []
    
    try:
        import gradio
    except ImportError:
        missing_deps.append("gradio")
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")
    
    if missing_deps:
        log_message(f"Missing dependencies: {', '.join(missing_deps)}", "ERROR")
        log_message("Please install: pip install -r requirements.txt", "ERROR")
        return False
    
    # Set up sequential downloads
    setup_sequential_downloads()
    
    return True

def check_port_available(port: int) -> bool:
    """Check if port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except OSError:
            return False

def launch_ui():
    """Launch Gradio UI with comprehensive error handling."""
    log_message("🚀 Starting Ethereal Canvas...")
    
    # Apply CUDA fixes FIRST before any imports
    try:
        setup_cuda_environment()
    except Exception as e:
        log_message(f"⚠️ CUDA setup failed, continuing anyway: {e}")
    
    # Set up sequential downloads
    setup_sequential_downloads()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set up paths
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Check port availability
    default_port = 7860
    port = default_port
    if not check_port_available(port):
        log_message(f"Port {port} is busy, trying alternatives...")
        for alt_port in range(7861, 7870):
            if check_port_available(alt_port):
                port = alt_port
                log_message(f"Using port {port}")
                break
        else:
            log_message("No available ports found", "ERROR")
            sys.exit(1)
    
    try:
        # Import and launch UI
        log_message("Initializing UI components...")
        from modules.ui_gradio.ui import launch_ui
        
        log_message("Launching Gradio interface...")
        launch_ui(
            server_name="0.0.0.0",
            server_port=port,
            share=False
        )
        
    except ImportError as e:
        log_message(f"Failed to import UI components: {e}", "ERROR")
        log_message("Please check that all modules are properly installed", "ERROR")
        sys.exit(1)
        
    except Exception as e:
        log_message(f"Failed to launch UI: {e}", "ERROR")
        log_message("Full error traceback:", "ERROR")
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main function."""
    try:
        launch_ui()
    except KeyboardInterrupt:
        log_message("👋 Shutting down Ethereal Canvas...")
    except Exception as e:
        log_message(f"Unexpected error: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()