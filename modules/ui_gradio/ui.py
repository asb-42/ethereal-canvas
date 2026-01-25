"""
Ethereal Canvas Gradio UI with separate T2I and I2I flows.
Real-time logging, status updates, and backend integration.
"""

import gradio as gr
import os
import sys
from pathlib import Path
from datetime import datetime
import traceback
import threading

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our backend system
from modules.backends.adapter import BackendAdapter
from modules.job_runner.runner_simple import execute_task, get_model_info

# Import pipeline monitoring
try:
    from utils.pipeline_monitor import ui_status_logger
    MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: Pipeline monitoring not available for UI")
    MONITORING_AVAILABLE = False


class EtherealCanvasUI:
    """Main UI class for Ethereal Canvas."""
    
    def __init__(self):
        """Initialize UI components."""
        self.backend_adapter = None
        self.is_processing = False
        self.status_timer = None
        
        # Simple backend initialization
        try:
            import yaml
            with open("config/model_config.yaml") as f:
                config = yaml.safe_load(f)
        except:
            config = {
                'generate_model': 'Qwen/Qwen-Image-2512',
                'edit_model': 'Qwen/Qwen-Image-Edit-2511'
            }
        
        self.backend_adapter = BackendAdapter(config)
    
    def get_status_updates(self):
        """Get recent status updates from pipeline monitor."""
        if MONITORING_AVAILABLE:
            recent_messages = ui_status_logger.get_recent_messages(20)
            return "\\n".join(recent_messages)
        return "No monitoring available"
    
    def start_status_updates(self):
        """Start periodic status updates using threading."""
        if self.status_timer:
            return
        
        def update_status():
            try:
                if hasattr(self, 't2i_log_component') and MONITORING_AVAILABLE:
                    status_text = self.get_status_updates()
                    self.t2i_log_component.value = status_text
            except Exception as e:
                print(f"Status update error: {e}")
            
            # Schedule next update
            if self.status_timer:
                self.status_timer = threading.Timer(2.0, update_status)
                self.status_timer.daemon = True
                self.status_timer.start()
    
    def stop_status_updates(self):
        """Stop status updates."""
        if self.status_timer:
            self.status_timer.cancel()
            self.status_timer = None
    
    def _log_message(self, message: str, level: str = "INFO") -> str:
        """Format log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] {level}: {message}"
    
    def generate_t2i(self, prompt: str, seed: int | None = None):
        """Generate image from text prompt."""
        if self.is_processing:
            return None, self._log_message("⚠️ Another task is running. Please wait...", "ERROR"), "error"
        
        if not prompt.strip():
            return None, self._log_message("Please enter a prompt", "ERROR"), "error"
        
        self.is_processing = True
        start_time = datetime.now()
        
        try:
            self._log_message(f"🎨 Starting T2I generation: {prompt[:50]}...")
            
            # Execute T2I task using job runner
            result = execute_task(
                task_type="generate",
                prompt_text=prompt,
                seed=seed,
                backend_adapter=self.backend_adapter
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result and result != "error":
                success_msg = self._log_message(f"✅ T2I generation completed: {result}", "SUCCESS")
                success_msg += f"\\n⏱️ Duration: {duration:.1f}s"
                return result, success_msg, "success"
            else:
                error_msg = self._log_message(f"T2I generation failed: {result}", "ERROR")
                return None, error_msg, "error"
                
        except Exception as e:
            error_msg = self._log_message(f"T2I generation failed: {str(e)}", "ERROR")
            return None, error_msg, "error"
            
        finally:
            self.is_processing = False
    
    def edit_i2i(self, image_file, prompt: str, seed: int | None = None):
        """Edit image based on prompt."""
        if self.is_processing:
            return None, self._log_message("⚠️ Another task is running. Please wait...", "ERROR"), "error"
        
        if image_file is None:
            return None, self._log_message("Please upload an image to edit", "ERROR"), "error"
        
        if not prompt.strip():
            return None, self._log_message("Please enter an edit prompt", "ERROR"), "error"
        
        self.is_processing = True
        start_time = datetime.now()
        
        try:
            self._log_message(f"🖼️ Starting I2I editing: {prompt[:50]}...")
            
            # Execute I2I task using job runner
            result = execute_task(
                task_type="edit",
                prompt_text=prompt,
                input_path=image_file,
                seed=seed,
                backend_adapter=self.backend_adapter
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if result and result != "error":
                success_msg = self._log_message(f"✅ I2I editing completed: {result}", "SUCCESS")
                success_msg += f"\\n⏱️ Duration: {duration:.1f}s"
                return result, success_msg, "success"
            else:
                error_msg = self._log_message(f"I2I editing failed: {result}", "ERROR")
                return None, error_msg, "error"
                
        except Exception as e:
            error_msg = self._log_message(f"I2I editing failed: {str(e)}", "ERROR")
            return None, error_msg, "error"
            
        finally:
            self.is_processing = False
    
    def get_system_info(self):
        """Get system and backend information."""
        try:
            if self.backend_adapter:
                model_info = self.backend_adapter.get_model_info()
                backend_status = "✅ Ready"
            else:
                model_info = "Not loaded (using stub mode)"
                backend_status = "⚠️ Stub Mode"
            
            return {
                "Backend Status": backend_status,
                "Models": str(model_info),
                "Processing": "🔄 Busy" if self.is_processing else "✅ Idle"
            }
        except Exception as e:
            return {
                "Backend Status": f"❌ Error: {str(e)}",
                "Models": "Unknown",
                "Processing": "❓ Unknown"
            }
    
    def create_ui(self):
        """Create the Gradio UI."""
        # Create demo without theme/css in constructor (move to launch for Gradio 6.0+)
        with gr.Blocks(
            title="Ethereal Canvas - AI Image Generation & Editing"
        ) as demo:
            with gr.Tabs() as tabs:
                # Tab 1: Generate (T2I)
                with gr.TabItem("🎨 Generate", id="t2i"):
                    gr.Markdown("### Text-to-Image Generation")
                    gr.Markdown("Generate images from text using Qwen-Image-2512")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            t2i_prompt = gr.Textbox(
                                label="Generation Prompt",
                                placeholder="Describe the image you want to generate...",
                                lines=4,
                                max_lines=8
                            )
                            
                            with gr.Row():
                                t2i_generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                                t2i_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                            
                            with gr.Accordion("⚙️ Advanced Options", open=False):
                                t2i_seed = gr.Number(
                                    label="Seed (optional)",
                                    value=None,
                                    precision=0
                                )
                        
                        with gr.Column(scale=1):
                            t2i_output = gr.Image(
                                label="Generated Image",
                                type="filepath",
                                height=400
                            )
                            
                            t2i_log = gr.Textbox(
                                label="Status Log",
                                lines=5,
                                max_lines=10,
                                interactive=False,
                                elem_classes=["log-box"]
                            )
                            
                            t2i_info = gr.JSON(label="System Info", visible=False)
                            
                            t2i_download = gr.File(
                                label="Download Image",
                                visible=False
                            )
                    
                    # T2I Tab event handlers
                    t2i_generate_btn.click(
                        fn=self.generate_t2i,
                        inputs=[t2i_prompt, t2i_seed],
                        outputs=[t2i_output, t2i_log, t2i_download, t2i_generate_btn]
                    )
                    
                    t2i_clear_btn.click(
                        fn=lambda: ("", None, None),
                        outputs=[t2i_prompt, t2i_seed]
)
                
                # Tab 2: Edit (I2I)
                with gr.TabItem("✏️ Edit", id="edit"):
                    gr.Markdown("### Image-to-Image Editing")
                    gr.Markdown("Edit existing images using Qwen-Image-Edit-2511")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            input_image = gr.Image(
                                label="Upload Image",
                                type="filepath",
                                height=200
                            )
                            
                            edit_prompt = gr.Textbox(
                                label="Edit Prompt",
                                placeholder="Describe changes you want to make...",
                                lines=3,
                                max_lines=5
                            )
                            
                            with gr.Row():
                                edit_generate_btn = gr.Button("✏️ Edit Image", variant="primary", size="lg")
                                edit_clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                            
                            with gr.Accordion("⚙️ Advanced Options", open=False):
                                edit_seed = gr.Number(
                                    label="Seed (optional)",
                                    value=None,
                                    precision=0,
                                    info="Leave empty for random seed"
                                )
                        
                        with gr.Column(scale=2):
                            edit_output = gr.Image(
                                label="Edited Image",
                                type="filepath",
                                height=300
                            )
                            
                            edit_log = gr.Textbox(
                                label="Status Log",
                                lines=5,
                                max_lines=10,
                                interactive=False,
                                elem_classes=["log-box"]
                            )
                            
                            edit_info = gr.JSON(label="System Info", visible=False)
                            
                            edit_download = gr.File(
                                label="Download Image",
                                visible=False
                            )
                    
                    # Note: Real-time status updates handled by background threading
                    
                    # I2I Tab event handlers
                    edit_generate_btn.click(
                        fn=self.edit_i2i,
                        inputs=[input_image, edit_prompt, edit_seed],
                        outputs=[edit_output, edit_log, edit_download, edit_generate_btn]
                    )
                    
                    edit_clear_btn.click(
                        fn=lambda: ("", None, None),
                        outputs=[input_image, edit_prompt, edit_seed]
                    )
                
        return demo


def launch_ui(server_name="0.0.0.0", server_port=7860, share=False):
    """Launch Gradio UI."""
    ui = EtherealCanvasUI()
    ui.start_status_updates()
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
        css="""
        .log-box { font-family: monospace; font-size: 12px; }
        .gradio-container { max-width: 1200px; }
        """
    )