"""
Simple Gradio UI implementation matching specification.
"""

import gradio as gr
from modules.backends.adapter import BackendAdapter

# Initialize adapter
adapter = BackendAdapter({
    'generate_model': 'Qwen/Qwen-Image-2512',
    'edit_model': 'Qwen/Qwen-Image-Edit-2511'
})

def generate(prompt, seed):
    """Generate image from text prompt."""
    path = adapter.generate(prompt)
    return path, f"Generated: {path}"

def edit(image, prompt, seed):
    """Edit image based on prompt."""
    path = adapter.edit(prompt, image.name if hasattr(image, 'name') else str(image))
    return path, f"Edited: {path}"

with gr.Blocks() as app:
    with gr.Tab("Generate"):
        p = gr.Textbox(label="Prompt")
        s = gr.Number(label="Seed", value=None)
        b = gr.Button("Generate")
        o = gr.Image()
        l = gr.Textbox(label="Log")
        b.click(generate, [p, s], [o, l])

    with gr.Tab("Edit"):
        i = gr.File(label="Input Image")
        p = gr.Textbox(label="Edit Prompt")
        s = gr.Number(label="Seed", value=None)
        b = gr.Button("Edit")
        o = gr.Image()
        l = gr.Textbox(label="Log")
        b.click(edit, [i, p, s], [o, l])

app.launch(server_name="0.0.0.0", server_port=7860)