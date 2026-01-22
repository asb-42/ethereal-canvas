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

def log_message(message: str, level: str = "INFO"):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

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
    """Launch the Gradio UI with comprehensive error handling."""
    log_message("🚀 Starting Ethereal Canvas...")
    
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