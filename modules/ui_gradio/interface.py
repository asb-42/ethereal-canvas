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
from modules.prompt_engine.templates import PROMPT_TEMPLATES, apply_template

def launch_ui(run_generate, run_edit, get_logs, config):
    with gr.Blocks() as demo:
        gr.Markdown("# Ethereal Canvas")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Upload Image (optional)")
                prompt_text = gr.Textbox(label="Prompt", placeholder="Enter prompt here")
                seed_input = gr.Number(label="Seed (optional)", value=0)
                prompt_style = gr.Dropdown(
                    choices=["none"] + list(PROMPT_TEMPLATES.keys()),
                    value="none",
                    label="Prompt Style"
                )
                enable_inpainting = gr.Checkbox(label="Enable Inpainting (coming soon)", value=False, interactive=False)
                button_generate = gr.Button("Generate")
                button_edit = gr.Button("Edit")
                output_image = gr.Image(label="Output Image")
                download_btn = gr.Button("Download")

        log_display = gr.Textbox(label="Logs", interactive=False)

        def apply_prompt_style(prompt, style):
            if style == "none":
                return prompt
            return apply_template(style, prompt)

        # Update prompt when style changes
        prompt_style.change(
            apply_prompt_style,
            inputs=[prompt_text, prompt_style],
            outputs=[prompt_text]
        )

        button_generate.click(
            lambda p, s, style: run_generate(apply_prompt_style(p, style), int(s) if s else None),
            inputs=[prompt_text, seed_input, prompt_style],
            outputs=[output_image, log_display]
        )

        button_edit.click(
            lambda p, s, img, style: run_edit(apply_prompt_style(p, style), int(s) if s else None, img),
            inputs=[prompt_text, seed_input, input_image, prompt_style],
            outputs=[output_image, log_display]
        )

        download_btn.click(lambda out: out, inputs=[output_image], outputs=[gr.File()])
        
        demo.launch(
            server_name=config["host"],
            server_port=config["port"]
        )