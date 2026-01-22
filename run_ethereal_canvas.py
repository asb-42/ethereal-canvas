#!/usr/bin/env python3
"""
Ethereal Canvas - Main Application Launcher
Entry point for Gradio UI application.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

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
    
    if venv_python:
        # Use absolute paths for comparison to handle symlinks properly
        venv_python_abs = os.path.abspath(venv_python)
        sys_executable_abs = os.path.abspath(sys.executable)
        
        if venv_python_abs != sys_executable_abs:
            print(f"🔄 Switching to virtual environment: {venv_python}")
            try:
                os.execv(venv_python, [venv_python] + sys.argv)
            except OSError as e:
                print(f"❌ Failed to execute virtual environment: {e}")
                print(f"🐍 Falling back to system Python: {sys.executable}")
        else:
            print(f"✅ Already running in virtual environment: {venv_python}")
    
    elif not os.path.exists(".venv"):
        print("⚠️  Virtual environment not found. Creating one...")
        try:
            subprocess.run(["python3", "-m", "venv", ".venv"], 
                          capture_output=True, text=True, check=True)
            print("✅ Virtual environment created successfully.")
            print("📋 Please run the following commands:")
            print("  source .venv/bin/activate")
            print("  python run_ethereal_canvas.py")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            sys.exit(1)
    
    # If we reach here, we're either in the right Python or venv setup failed
    # Try to launch a working version of the application
    try:
        print("🚀 Launching Ethereal Canvas...")
        import gradio as gr
        print("✅ Gradio version:", gr.__version__)
        
        # Test basic functionality first
        def simple_generate(prompt):
            return f"Generated: {prompt}"
        
        def simple_edit(prompt):
            return f"Edited: {prompt}"
        
        # Create minimal UI that works
        with gr.Blocks(title="Ethereal Canvas - AI Image Generation & Editing") as demo:
            gr.Markdown("# 🎨 Ethereal Canvas\nAI-powered image generation and editing")
            
            with gr.Tabs():
                with gr.Tab("🖼️ Generate"):
                    prompt_input = gr.Textbox(label="Prompt", placeholder="Enter your image description...")
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                    output = gr.Textbox(label="Result", interactive=False)
                    
                    generate_btn.click(simple_generate, inputs=prompt_input, outputs=output)
                
                with gr.Tab("✏️ Edit"):
                    edit_prompt = gr.Textbox(label="Edit Prompt", placeholder="Describe the changes...")
                    edit_btn = gr.Button("✏️ Edit Image", variant="primary")
                    edit_output = gr.Textbox(label="Result", interactive=False)
                    
                    edit_btn.click(simple_edit, inputs=edit_prompt, outputs=edit_output)
        
        print("🌐 Starting server on http://localhost:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📋 Please install dependencies:")
        print("  source .venv/bin/activate  # if using venv")
        print("  pip install gradio torch transformers")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()