#!/usr/bin/env python3
"""
Minimal working Ethereal Canvas launcher.
"""

import os
import sys
from pathlib import Path

def main():
    print("🎨 Ethereal Canvas - AI Image Generation & Editing")
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Virtual environment check
    if os.path.exists(".venv/bin/python"):
        print("🔄 Using virtual environment...")
        # Re-run with virtual environment
        os.execv(".venv/bin/python", [__file__] + sys.argv)
    
    # Simple UI test
    try:
        import gradio as gr
        print("✅ Gradio version:", gr.__version__)
        
        # Test basic functionality
        def simple_generate(prompt):
            return f"Generated: {prompt}"
        
        def simple_edit(prompt):
            return f"Edited: {prompt}"
        
        # Create minimal UI
        with gr.Blocks(title="Ethereal Canvas - Test") as demo:
            gr.Markdown("# 🎨 Ethereal Canvas - Test Interface")
            
            with gr.Tab("Generate"):
                prompt_input = gr.Textbox(label="Prompt", placeholder="Enter text...")
                generate_btn = gr.Button("Generate")
                output = gr.Textbox(label="Result", interactive=False)
                
                generate_btn.click(simple_generate, inputs=prompt_input, outputs=output)
            
            with gr.Tab("Edit"):
                edit_prompt = gr.Textbox(label="Edit Prompt", placeholder="Enter edit...")
                edit_btn = gr.Button("Edit") 
                edit_output = gr.Textbox(label="Result", interactive=False)
                
                edit_btn.click(simple_edit, inputs=edit_prompt, outputs=edit_output)
        
        print("🚀 Starting test server on http://localhost:7860")
        demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📋 Please run: bash scripts/install_pinokio.sh")
        return 1
    
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())