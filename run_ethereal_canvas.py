#!/usr/bin/env python3
"""
Ethereal Canvas - Main Application Launcher
Entry point for Gradio UI application.
"""

import os
import sys
from pathlib import Path

def main():
    """Main entry point for the application."""
    print("🎨 Ethereal Canvas - AI Image Generation & Editing")
    print("=" * 50)
    
    # Check if virtual environment exists and use it
    venv_python = None
    venv_paths = [
        ".venv/bin/python3",
        ".venv/bin/python",
        "venv/bin/python3", 
        "venv/bin/python"
    ]
    
    for venv_path in venv_paths:
        if os.path.exists(venv_path):
            venv_python = venv_path
            break
    
    if venv_python and venv_python != sys.executable:
        print(f"🔄 Switching to virtual environment: {venv_python}")
        try:
            os.execv(venv_python, [venv_python] + sys.argv)
        except OSError as e:
            print(f"❌ Failed to execute virtual environment: {e}")
            print(f"🐍 Falling back to system Python: {sys.executable}")
    
    elif not os.path.exists(".venv"):
        print("⚠️  Virtual environment not found. Creating one...")
        import subprocess
        result = subprocess.run(["python3", "-m", "venv", ".venv"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to create virtual environment: {result.stderr}")
            sys.exit(1)
        print("✅ Virtual environment created successfully.")
        print("📋 Please run the following commands:")
        print("  source .venv/bin/activate")
        print("  python run_ethereal_canvas.py")
        sys.exit(0)
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    try:
        # Import and launch the UI
        from modules.ui_gradio.simple_ui import launch_ui
        
        # Get configuration from environment or use defaults
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        
        print(f"🚀 Starting server on {server_name}:{server_port}")
        print(f"🌐 Share: {share}")
        
        # Launch UI
        launch_ui(
            server_name=server_name,
            server_port=server_port,
            share=share
        )
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()