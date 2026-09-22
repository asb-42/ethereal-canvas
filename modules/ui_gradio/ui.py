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
        # Read deliberately outside the try below: a configuration that cannot
        # be read must stop the launch, not be swallowed into the generic
        # "Failed to initialize backend" tuple that __init__ discards.
        from modules.runtime.paths import load_model_config
        config = load_model_config()

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
    
    def generate_t2i(self, prompt: str, seed: int | None = None,
                     text_encoder: str | None = None,
                     width: int | None = None, height: int | None = None,
                     prompt_rewrite: str | None = None,
                     progress_cb=None):
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
            size_kwargs = {}
            if width and height:
                size_kwargs = {"width": int(width), "height": int(height)}
            opt_kwargs = dict(size_kwargs)
            if text_encoder:
                opt_kwargs["text_encoder"] = text_encoder
            if prompt_rewrite:
                opt_kwargs["prompt_rewrite"] = prompt_rewrite
            if progress_cb is not None:
                opt_kwargs["progress_cb"] = progress_cb
            if self.backend_adapter:
                print("🔍 Testing backend adapter...")
                result = self.backend_adapter.generate(prompt, **opt_kwargs)
                print(f"🔍 Backend result: {result}")
            else:
                print("🔍 Using task runner fallback...")
                # Fallback to simple task runner
                if seed is not None and seed > 0:
                    result = execute_task("generate", prompt, seed=seed,
                                          text_encoder=text_encoder,
                                          prompt_rewrite=prompt_rewrite,
                                          **size_kwargs)
                else:
                    result = execute_task("generate", prompt,
                                          text_encoder=text_encoder,
                                          prompt_rewrite=prompt_rewrite,
                                          **size_kwargs)
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
    
    def edit_i2i(self, image_file, prompt: str, seed: int | None = None,
                 text_encoder: str | None = None,
                 output_resolution: int | None = None,
                 progress_cb=None):
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
            enc_kwargs = ({"text_encoder": text_encoder} if text_encoder else {})
            res_kwargs = ({"output_resolution": int(output_resolution)}
                          if output_resolution else {})
            if progress_cb is not None:
                res_kwargs["progress_cb"] = progress_cb
            if self.backend_adapter:
                result = self.backend_adapter.edit(
                    prompt, image_path, **enc_kwargs, **res_kwargs)
            else:
                # Fallback to simple task runner
                if seed is not None and seed > 0:
                    result = execute_task("edit", prompt, input_path=image_path, seed=seed,
                                          text_encoder=text_encoder,
                                          **res_kwargs)
                else:
                    result = execute_task("edit", prompt, input_path=image_path,
                                          text_encoder=text_encoder,
                                          **res_kwargs)
            
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
                # Bind the component to a name so it can serve as an output target.
                # The previous code kept only the data dict in `system_info` and then
                # passed that dict as demo.load(outputs=...), which made Gradio crash
                # later while building its config. The value stays the plain info dict,
                # which this component renders natively.
                system_info = gr.JSON(
                    value=self.get_system_info(),
                    label="Backend Information",
                )
            
            # Main tabs
            with gr.Tabs():
                
                # Tab 1: Generate (T2I)
                with gr.TabItem("🖼️ Generate", id="generate"):
                    gr.Markdown("### Text-to-Image Generation")
                    gr.Markdown(f"Generate images from text descriptions using {self.backend_adapter.t2i_model if self.backend_adapter else 'Qwen-Image-2512'}")
                    
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

                                t2i_encoder = gr.Dropdown(
                                    label="Text encoder",
                                    choices=["Stock", "Heretic (abliterated)"],
                                    value="Stock",
                                    info="Heretic removes refusal behaviour; needs its weights on disk"
                                )

                                t2i_size = gr.Dropdown(
                                    label="Image size",
                                    choices=["Auto (rewriter recommends, else 1024²)",
                                             "512 × 512 (~13s)",
                                             "768 × 768 (~30s)",
                                             "1024 × 1024 default (~1 min)",
                                             "2048 × 2048 card recommended (~4 min)",
                                             "16:9 wide (~4 min)",
                                             "9:16 tall (~4 min)"],
                                    value="1024 × 1024 default (~1 min)",
                                    info="Times measured on this box at 40 steps"
                                )

                                t2i_rewrite = gr.Checkbox(
                                    label="Expand prompt (PE-T2I rewriter)",
                                    value=False,
                                    info="18 GiB companion, reloaded per run to protect VRAM"
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
                    gr.Markdown(f"Edit existing images using {self.backend_adapter.edit_model if self.backend_adapter else 'Qwen-Image-Edit-2511'}")
                    
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

                                edit_encoder = gr.Dropdown(
                                    label="Text encoder",
                                    choices=["Stock", "Heretic (abliterated)"],
                                    value="Stock",
                                    info="Heretic removes refusal behaviour; needs its weights on disk"
                                )

                                edit_size = gr.Dropdown(
                                    label="Output resolution",
                                    choices=["1024 (faster)",
                                             "2048 default (card recommended)"],
                                    value="2048 default (card recommended)",
                                    info="Single 2048 edit measured ~7 min on this box"
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
            _t2i = self.backend_adapter.t2i_model if self.backend_adapter else "Qwen-Image-2512"
            _edit = self.backend_adapter.edit_model if self.backend_adapter else "Qwen-Image-Edit-2511"
            gr.Markdown(f"""
            ---
            **Models**: {_t2i} (Generation) | {_edit} (Editing)
            """)
            
            # Event handlers
            def _variant(label):
                return "heretic" if label and "heretic" in label.lower() else "stock"

            #: UI size labels to explicit dimensions. Fixed presets, not free
            #: numbers: the pipeline only warns and resizes dimensions that are
            #: not divisible by 32, so every preset here is pre-divisible.
            T2I_SIZES = {
                "512": (512, 512),
                "768": (768, 768),
                "1024": (1024, 1024),
                "2048": (2048, 2048),
                "16:9": (2752, 1536),
                "9:16": (1536, 2752),
            }

            def _t2i_size(label):
                if label and label.startswith("Auto"):
                    return (None, None)
                for key, size in T2I_SIZES.items():
                    if label and label.startswith(key):
                        return size
                return (1024, 1024)

            def _edit_size(label):
                if label and label.startswith("1024"):
                    return 1024
                return 2048

            def handle_generate(prompt, seed, encoder_label, size_label, rewrite_on):
                """Handle generate button click, streaming step progress."""
                import time
                width, height = _t2i_size(size_label)
                variant = _variant(encoder_label)
                rw = "on" if rewrite_on else None
                state = {"step": 0, "total": 40}
                box = {}

                def work():
                    box["out"] = self.generate_t2i(
                        prompt, seed, text_encoder=variant,
                        width=width, height=height, prompt_rewrite=rw,
                        progress_cb=lambda s, t: state.update(step=s, total=t))

                th = threading.Thread(target=work, daemon=True)
                th.start()
                while th.is_alive():
                    s, t = state["step"], state["total"]
                    if s > 0:
                        msg = f"Denoising step {s}/{t} ..."
                    else:
                        msg = "Loading model / expanding prompt ..."
                    yield (None, self._log_message(msg, "INFO"),
                           gr.update(visible=False),
                           gr.update(interactive=False))
                    time.sleep(2)
                th.join()

                image_path, log_msg, status = box["out"]
                if status == "success" and image_path and os.path.exists(image_path):
                    yield (
                        image_path,           # image
                        log_msg,              # log
                        gr.update(value=image_path, visible=True),  # download
                        gr.update(interactive=False)  # disable button
                    )
                else:
                    yield (
                        None,                 # image
                        log_msg,              # log
                        gr.update(visible=False),           # download
                        gr.update(interactive=False)          # disable button
                    )
            
            def handle_edit(image, prompt, seed, encoder_label, size_label):
                """Handle edit button click, streaming step progress."""
                import time
                variant = _variant(encoder_label)
                resolution = _edit_size(size_label)
                state = {"step": 0, "total": 40}
                box = {}

                def work():
                    box["out"] = self.edit_i2i(
                        image, prompt, seed, text_encoder=variant,
                        output_resolution=resolution,
                        progress_cb=lambda s, t: state.update(step=s, total=t))

                th = threading.Thread(target=work, daemon=True)
                th.start()
                while th.is_alive():
                    s, t = state["step"], state["total"]
                    if s > 0:
                        msg = f"Denoising step {s}/{t} ..."
                    else:
                        msg = "Loading model / preparing edit ..."
                    yield (None, self._log_message(msg, "INFO"),
                           gr.update(visible=False),
                           gr.update(interactive=False))
                    time.sleep(2)
                th.join()

                image_path, log_msg, status = box["out"]
                if status == "success" and image_path and os.path.exists(image_path):
                    yield (
                        image_path,           # image
                        log_msg,              # log
                        gr.update(value=image_path, visible=True),  # download
                        gr.update(interactive=False)  # disable button
                    )
                else:
                    yield (
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
                inputs=[prompt_input, seed_input, t2i_encoder, t2i_size, t2i_rewrite],
                outputs=[t2i_output, t2i_log, t2i_download, generate_btn],
                show_progress="minimal"
            ).then(
                fn=reset_buttons,
                outputs=[generate_btn, edit_btn]
            )
            
            edit_btn.click(
                fn=handle_edit,
                inputs=[input_image, edit_prompt, edit_seed, edit_encoder, edit_size],
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
            # Refresh the panel on page load. Deliberately not wrapped in
            # try/except: the previous (AttributeError, TypeError) guard swallowed
            # the mis-wired outputs= above and let it surface later, as a crash
            # during config build.
            demo.load(fn=self.get_system_info, outputs=[system_info])
            
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