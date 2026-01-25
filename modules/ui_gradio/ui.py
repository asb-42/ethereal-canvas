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
        self.abort_requested = False

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

        try:
            self.backend_adapter = BackendAdapter(config)
            self.backend_adapter.load()
        except Exception as e:
            print(f"Backend initialization error: {e}")
            # Continue with None backend - will work in stub mode
            self.backend_adapter = None
        
        # Start status updates after UI is created
        self.start_status_updates()
    
    def _log_message(self, message: str, status: str = "INFO"):
        """Format log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {status}: {message}"
    
    def generate_t2i(self, prompt: str, seed: int | None = None):
        """Generate image from text prompt."""
        if self.is_processing:
            return None, "⚠️ Another task is running. Please wait...", "error"
        
        self.is_processing = True
        self.abort_requested = False
        
        try:
            # Log start
            initial_msg = f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Starting T2I generation: {prompt[:50]}..."
            print(f"🔍 DEBUG: Initial message: {initial_msg}")
            
            # Directly update T2I log if component exists
            if hasattr(self, 't2i_log_component'):
                current_logs = self.t2i_log_component.value or ""
                updated_logs = current_logs + "\n" + initial_msg if current_logs else initial_msg
                self.t2i_log_component.value = updated_logs
                print("🔍 DEBUG: Updated T2I log with initial message")
            
            # Execute generation using adapter if available, otherwise fallback
            if self.backend_adapter:
                print("🔍 Testing backend adapter...")
                result = self.backend_adapter.generate(prompt)
                print(f"🔍 Backend result: {result}")
            else:
                print("🔍 Using task runner fallback...")
                # Fallback to simple task runner
                if seed is not None and seed > 0:
                    result = execute_task("generate", prompt, seed=seed)
                else:
                    result = execute_task("generate", prompt)
                print(f"🔍 Task runner result: {result}")
            
            # Get actual image path
            if result and os.path.exists(result):
                image_result = result
                success_msg = self._log_message(f"T2I generation completed: {result}", "SUCCESS")
            elif result and result.endswith('.png'):
                # Check if result is a valid path even if file doesn't exist yet (stub mode)
                image_result = result
                success_msg = self._log_message(f"T2I generation completed: {result}", "SUCCESS")
            else:
                # No valid image path
                image_result = None
                success_msg = self._log_message(f"T2I generation failed: {result}", "ERROR")
            
            return image_result, success_msg, "success"
            
        except Exception as e:
            error_msg = self._log_message(f"T2I generation failed: {str(e)}", "ERROR")
            return None, error_msg, "error"
            
        finally:
            self.is_processing = False
    
    def abort_generation(self):
        """Abort current generation process."""
        print("🔍 DEBUG: abort_generation method called!")
        self.abort_requested = True
        self.is_processing = False
        
        # Create abort flag file for backend to detect
        import os
        from pathlib import Path
        runtime_dir = Path("runtime")
        runtime_dir.mkdir(exist_ok=True)
        abort_file = runtime_dir / ".abort_generation"
        abort_file.touch()
        
        print(f"🔍 DEBUG: Abort file created at: {abort_file}")
        print(f"🔍 DEBUG: Abort file exists: {abort_file.exists()}")
        
        print("🛑 Generation/Editing aborted by user")
        abort_msg = self._log_message("Generation/Editing aborted by user", "INFO")
        print(f"🔍 DEBUG: Abort message: {abort_msg}")
        # Return outputs for both generate and edit abort buttons
        return (
            abort_msg,           # log message
            gr.update(interactive=True),  # enable generate/edit button
            gr.update(interactive=False)  # disable abort button
        )
    
    def edit_i2i(self, image_file, prompt: str, seed: int | None = None):
        """Edit image based on prompt."""
        if self.is_processing:
            return None, "⚠️ Another task is running. Please wait...", "error"
        
        if image_file is None:
            return None, self._log_message("Please upload an image to edit", "ERROR"), "error"
        
        self.is_processing = True
        self.abort_requested = False
        
        try:
            # Get uploaded image path
            image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
            
            # Log start
            initial_msg = f"[{datetime.now().strftime('%H:%M:%S')}] INFO: Starting I2I edit: {prompt[:50]}..."
            print(f"🔍 DEBUG: I2I initial message: {initial_msg}")
            
            # Directly update Edit log if component exists
            if hasattr(self, 'edit_log_component'):
                current_logs = self.edit_log_component.value or ""
                updated_logs = current_logs + "\n" + initial_msg if current_logs else initial_msg
                self.edit_log_component.value = updated_logs
                print("🔍 DEBUG: Updated Edit log with initial message")
            
            # Execute edit using adapter if available, otherwise fallback
            if self.backend_adapter:
                result = self.backend_adapter.edit(prompt, image_path)
            else:
                # Fallback to simple task runner
                if seed is not None and seed > 0:
                    result = execute_task("edit", prompt, input_path=image_path, seed=seed)
                else:
                    result = execute_task("edit", prompt, input_path=image_path)
            
            # Get actual image path
            if os.path.exists(result):
                image_result = result
                success_msg = self._log_message(f"I2I edit completed: {result}", "SUCCESS")
            else:
                # Fallback for stub mode
                image_result = None
                success_msg = self._log_message(f"I2I edit completed (stub mode): {result}", "SUCCESS")
            
            return image_result, success_msg, "success"
            
        except Exception as e:
            error_msg = self._log_message(f"I2I edit failed: {str(e)}", "ERROR")
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
    
    def get_status_updates(self):
        """Get recent status updates from pipeline monitor."""
        if MONITORING_AVAILABLE:
            recent_messages = ui_status_logger.get_recent_messages(20)
            return "\\n".join(recent_messages)
        return "No monitoring available"
    
    def start_status_updates(self):
        """Start periodic status updates."""
        if self.status_timer:
            return
        
        def update_status():
            try:
                if MONITORING_AVAILABLE:
                    status_text = self.get_status_updates()
                    print(f"🔍 DEBUG: Status update called with: {status_text[:100]}...")
                    
                    # Update both log components if they exist
                    if hasattr(self, 't2i_log_component'):
                        self.t2i_log_component.value = status_text
                        print("🔍 DEBUG: Updated T2I log component")
                    if hasattr(self, 'edit_log_component'):
                        self.edit_log_component.value = status_text
                        print("🔍 DEBUG: Updated Edit log component")
                else:
                    print("🔍 DEBUG: MONITORING_AVAILABLE is False")
                return
            except Exception as e:
                print(f"🔍 DEBUG: Status update error: {e}")
                import traceback
                traceback.print_exc()
        
        # Start timer for updates every 2 seconds
        self.status_timer = threading.Timer(2.0, update_status)
        self.status_timer.daemon = True
        self.status_timer.start()
    
    def stop_status_updates(self):
        """Stop status updates."""
        if self.status_timer:
            self.status_timer.cancel()
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
    
    def create_ui(self):
        """Create the Gradio UI."""
        # Create demo without theme/css in constructor (move to launch for Gradio 6.0+)
        with gr.Blocks(
            title="Ethereal Canvas - AI Image Generation & Editing"
        ) as demo:
            
            # Header
            gr.Markdown("""
            # 🎨 Ethereal Canvas
            AI-powered image generation and editing using Qwen models
            """)
            
            # System info
            with gr.Accordion("System Status", open=False):
                system_info = self.get_system_info()
                gr.JSON(value=system_info, label="Backend Information")
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Generate (T2I)
                with gr.TabItem("🖼️ Generate", id="generate"):
                    gr.Markdown("### Text-to-Image Generation")
                    gr.Markdown("Generate images from text descriptions using Qwen-Image-2512")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            prompt_input = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter your image description here...",
                                lines=3,
                                max_lines=5
                            )
                            
                            with gr.Row():
                                seed_input = gr.Number(
                                    label="Seed (optional)",
                                    value=None,
                                    precision=0,
                                    info="Leave empty for random seed"
                                )
                                
                                with gr.Row():
                                    generate_btn = gr.Button(
                                        "🎨 Generate Image",
                                        variant="primary",
                                        size="lg"
                                    )
                                    abort_generate_btn = gr.Button(
                                        "⏹️ Abort",
                                        variant="stop",
                                        size="lg"
                                    )
                        
                        with gr.Column(scale=2):
                            t2i_output = gr.Image(
                                label="Generated Image",
                                type="filepath",
                                height=300
                            )
                            
                            t2i_download = gr.File(
                                label="Download Image",
                                visible=False
                            )
                    
                    t2i_log = gr.Textbox(
                        label="Status Log",
                        lines=5,
                        max_lines=10,
                        interactive=False,
                        elem_classes=["log-box"]
                    )
                    
                    # Store reference for status updates
                    self.t2i_log_component = t2i_log
                
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
                                placeholder="Describe the changes you want to make...",
                                lines=3,
                                max_lines=5
                            )
                            
                            with gr.Row():
                                edit_seed = gr.Number(
                                    label="Seed (optional)",
                                    value=None,
                                    precision=0,
                                    info="Leave empty for random seed"
                                )
                                
                                with gr.Row():
                                    edit_btn = gr.Button(
                                        "✏️ Edit Image",
                                        variant="primary",
                                        size="lg"
                                    )
                                    abort_edit_btn = gr.Button(
                                        "⏹️ Abort",
                                        variant="stop",
                                        size="lg"
                                    )
                        
                        with gr.Column(scale=2):
                            edit_output = gr.Image(
                                label="Edited Image",
                                type="filepath",
                                height=300
                            )
                            
                            edit_download = gr.File(
                                label="Download Image",
                                visible=False
                            )
                    
                    edit_log = gr.Textbox(
                        label="Status Log",
                        lines=5,
                        max_lines=10,
                        interactive=False,
                        elem_classes=["log-box"]
                    )
                    
                    # Store reference for status updates
                    self.edit_log_component = edit_log
            
            # Footer
            gr.Markdown("""
            ---
            **Models**: Qwen-Image-2512 (Generation) | Qwen-Image-Edit-2511 (Editing)
            """)
            
            # Event handlers
            def handle_generate(prompt, seed):
                """Handle generate button click."""
                image_path, log_msg, status = self.generate_t2i(prompt, seed)
                
                if status == "success" and image_path and os.path.exists(image_path):
                    return (
                        image_path,           # image
                        log_msg,              # log
                        gr.update(value=image_path, visible=True),  # download
                        gr.update(interactive=False)  # disable button
                    )
                else:
                    return (
                        None,                 # image
                        log_msg,              # log
                        gr.update(visible=False),           # download
                        gr.update(interactive=False)          # disable button
                    )
            
            def handle_edit(image, prompt, seed):
                """Handle edit button click."""
                image_path, log_msg, status = self.edit_i2i(image, prompt, seed)
                
                if status == "success" and image_path and os.path.exists(image_path):
                    return (
                        image_path,           # image
                        log_msg,              # log
                        gr.update(value=image_path, visible=True),  # download
                        gr.update(interactive=False)  # disable button
                    )
                else:
                    return (
                        None,                 # image
                        log_msg,              # log
                        gr.update(visible=False),           # download
                        gr.update(interactive=False)          # disable button
                    )
            
            def reset_buttons():
                """Reset buttons to enabled state."""
                return (
                    gr.update(interactive=True),   # generate button
                    gr.update(interactive=True)    # edit button
                )
            
            # Wire up events
            generate_btn.click(
                fn=handle_generate,
                inputs=[prompt_input, seed_input],
                outputs=[t2i_output, t2i_log, t2i_download, generate_btn],
                show_progress="minimal"
            ).then(
                fn=reset_buttons,
                outputs=[generate_btn, edit_btn]
            )
            
            edit_btn.click(
                fn=handle_edit,
                inputs=[input_image, edit_prompt, edit_seed],
                outputs=[edit_output, edit_log, edit_download, edit_btn],
                show_progress="minimal"
            ).then(
                fn=reset_buttons,
                outputs=[generate_btn, edit_btn]
            )
            
            # Abort button events
            abort_generate_btn.click(
                fn=self.abort_generation,
                outputs=[t2i_log, generate_btn, abort_generate_btn]
            )
            
            abort_edit_btn.click(
                fn=self.abort_generation,
                outputs=[edit_log, edit_btn, abort_edit_btn]
            )
            
            # Initial system status update
            # Note: demo.load() moved to .ready() in newer Gradio versions
        try:
            demo.load(
                fn=self.get_system_info,
                outputs=system_info
            )
        except AttributeError:
            # Fallback for newer Gradio versions
            pass
        
        return demo

# Create and launch the UI
def launch_ui(server_name="0.0.0.0", server_port=7860, share=False):
    """Launch the Gradio UI."""
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
        inbrowser=True
    )

if __name__ == "__main__":
    launch_ui()