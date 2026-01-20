"""
Gradio UI for Ethereal Canvas.
Defines the interactive components:
- upload
- prompt input
- buttons
- download
- logs
"""

import gradio as gr
from modules.job_runner.runner import execute_task
from modules.prompt_engine.engine import normalize_prompt

def launch_ui(run_generate, run_edit, get_logs, config):
    with gr.Blocks() as demo:
        gr.Markdown("# Ethereal Canvas")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Upload Image (optional)")
                prompt_text = gr.Textbox(label="Prompt", placeholder="Enter prompt here")
                seed_input = gr.Number(label="Seed (optional)", value=0)
                enable_inpainting = gr.Checkbox(label="Enable Inpainting (coming soon)", value=False, interactive=False)
                button_generate = gr.Button("Generate")
                button_edit = gr.Button("Edit")
                output_image = gr.Image(label="Output Image")
                download_btn = gr.Button("Download")

        log_display = gr.Textbox(label="Logs", interactive=False)

        button_generate.click(
            lambda p, s: run_generate(p, int(s) if s else None),
            inputs=[prompt_text, seed_input],
            outputs=[output_image, log_display]
        )

        button_edit.click(
            lambda p, s, img: run_edit(p, int(s) if s else None, img),
            inputs=[prompt_text, seed_input, input_image],
            outputs=[output_image, log_display]
        )

        download_btn.click(lambda out: out, inputs=[output_image], outputs=[gr.File()])
        
        demo.launch(
            server_name=config["host"],
            server_port=config["port"]
        )