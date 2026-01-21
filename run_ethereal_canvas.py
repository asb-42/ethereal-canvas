#!/usr/bin/env python3
"""
Ethereal Canvas - Main Application Launcher
Entry point for the Gradio UI application.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the application."""
    print("🎨 Ethereal Canvas - AI Image Generation & Editing")
    print("=" * 50)
    
    try:
        # Import and launch the UI
        from modules.ui_gradio.ui import launch_ui
        
        # Get configuration from environment or use defaults
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
        
        print(f"🚀 Starting server on {server_name}:{server_port}")
        print(f"🌐 Share: {share}")
        
        # Launch the UI
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