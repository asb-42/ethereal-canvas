"""
Simplified Ethereal Canvas Gradio UI with fixed Gradio 6.0+ compatibility.
"""

import gradio as gr
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our backend system
from modules.backends.adapter import BackendAdapter
from modules.job_runner.runner_simple import execute_task, get_model_info

class SimpleEtherealCanvasUI:
    """Simplified UI class for Ethereal Canvas."""
    
    def __init__(self):
        """Initialize UI with backend adapter."""
        self.backend_adapter = None
        self.is_processing = False
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize backend adapter."""
        try:
            import yaml
            with open("config/model_config.yaml") as f:
                config = yaml.safe_load(f)
        except:
            config = {
                'generate_model': 'Qwen/Qwen-Image-2512',
                'edit_model': 'Qwen/Qwen-Image-Edit-2511'
            }
        
        try:
            self.backend_adapter = BackendAdapter(config)
            self.backend_adapter.load()
            return True, "Backend initialized successfully"
        except Exception as e:
            return False, f"Failed to initialize backend: {str(e)}"
    
    def generate_t2i(self, prompt: str, seed: int = None):
        """Generate image from text prompt."""
        if self.is_processing:
            return None, "⚠️ Another task is running. Please wait..."
        
        self.is_processing = True
        
        try:
            if seed is not None and seed > 0:
                result = execute_task("generate", prompt, seed=seed)
            else:
                result = execute_task("generate", prompt)
            
            log_msg = f"✅ Generation completed: {result}"
            return result, log_msg
            
        except Exception as e:
            error_msg = f"❌ Generation failed: {str(e)}"
            return None, error_msg
            
        finally:
            self.is_processing = False
    
    def edit_i2i(self, image_file, prompt: str, seed: int = None):
        """Edit image based on prompt."""
        if self.is_processing:
            return None, "⚠️ Another task is running. Please wait..."
        
        if image_file is None:
            return None, "❌ Please upload an image to edit"
        
        self.is_processing = True
        
        try:
            # Get uploaded image path
            image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
            
            if seed is not None and seed > 0:
                result = execute_task("edit", prompt, input_path=image_path, seed=seed)
            else:
                result = execute_task("edit", prompt, input_path=image_path)
            
            log_msg = f"✅ Edit completed: {result}"
            return result, log_msg
            
        except Exception as e:
            error_msg = f"❌ Edit failed: {str(e)}"
            return None, error_msg
            
        finally:
            self.is_processing = False
    
    def get_system_info(self):
        """Get system and backend information."""
        try:
            model_info = get_model_info() if self.backend_adapter else {}
            return {
                "Backend Status": "✅ Ready" if self.backend_adapter else "❌ Failed",
                "Models": str(model_info) if model_info else "Not loaded",
                "Processing": "🔄 Busy" if self.is_processing else "✅ Idle"
            }
        except Exception as e:
            return {
                "Backend Status": f"❌ Error: {str(e)}",
                "Models": "Unknown",
                "Processing": "❓ Unknown"
            }
    
    def create_ui(self):
        """Create Gradio UI."""
        with gr.Blocks(title="Ethereal Canvas - AI Image Generation & Editing") as demo:
            # Header
            gr.Markdown("# 🎨 Ethereal Canvas\nAI-powered image generation and editing using Qwen models")
            
            # System info
            with gr.Accordion("System Status", open=False):
                system_info = self.get_system_info()
                gr.JSON(value=system_info, label="Backend Information")
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Generate (T2I)
                with gr.Tab("🖼️ Generate"):
                    gr.Markdown("### Text-to-Image Generation\nGenerate images from text descriptions using Qwen-Image-2512")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter your image description here...",
                                lines=3
                            )
                            
                            with gr.Row():
                                seed_input = gr.Number(
                                    label="Seed (optional)",
                                    value=None,
                                    precision=0
                                )
                                
                                generate_btn = gr.Button(
                                    "🎨 Generate Image",
                                    variant="primary"
                                )
                        
                        with gr.Column(scale=1):
                            t2i_output = gr.Image(
                                label="Generated Image",
                                type="filepath"
                            )
                    
                    t2i_log = gr.Textbox(
                        label="Status Log",
                        lines=5,
                        interactive=False
                    )
                
                # Tab 2: Edit (I2I)
                with gr.Tab("✏️ Edit"):
                    gr.Markdown("### Image-to-Image Editing\nEdit existing images using Qwen-Image-Edit-2511")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            input_image = gr.Image(
                                label="Upload Image",
                                type="filepath"
                            )
                            
                            edit_prompt = gr.Textbox(
                                label="Edit Prompt",
                                placeholder="Describe the changes you want to make...",
                                lines=3
                            )
                            
                            with gr.Row():
                                edit_seed = gr.Number(
                                    label="Seed (optional)",
                                    value=None,
                                    precision=0
                                )
                                
                                edit_btn = gr.Button(
                                    "✏️ Edit Image",
                                    variant="primary"
                                )
                        
                        with gr.Column(scale=1):
                            edit_output = gr.Image(
                                label="Edited Image",
                                type="filepath"
                            )
                    
                    edit_log = gr.Textbox(
                        label="Status Log",
                        lines=5,
                        interactive=False
                    )
            
            # Footer
            gr.Markdown("---\n**Models**: Qwen-Image-2512 (Generation) | Qwen-Image-Edit-2511 (Editing)")
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_t2i,
                inputs=[prompt_input, seed_input],
                outputs=[t2i_output, t2i_log]
            )
            
            edit_btn.click(
                fn=self.edit_i2i,
                inputs=[input_image, edit_prompt, edit_seed],
                outputs=[edit_output, edit_log]
            )
        
        return demo

def launch_ui(server_name="0.0.0.0", server_port=7860, share=False):
    """Launch the Gradio UI."""
    ui = SimpleEtherealCanvasUI()
    demo = ui.create_ui()
    
    print(f"🚀 Launching Ethereal Canvas UI...")
    print(f"📍 Server: http://{server_name}:{server_port}")
    if share:
        print("🌐 Creating public share link...")
    
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        show_error=True,
        inbrowser=True
    )