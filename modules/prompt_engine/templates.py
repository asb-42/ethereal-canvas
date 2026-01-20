"""
Prompt templates for consistent styling.
"""

PROMPT_TEMPLATES = {
    "photorealistic": {
        "suffix": "highly detailed, photorealistic, 8k, cinematic lighting"
    },
    "concept_art": {
        "suffix": "concept art, painterly, dramatic lighting"
    },
    "sketch": {
        "suffix": "pencil sketch, monochrome, rough lines"
    },
    "anime": {
        "suffix": "anime style, cel shading, vibrant colors"
    },
    "oil_painting": {
        "suffix": "oil painting, canvas texture, brush strokes visible"
    }
}

def apply_template(template_name: str, base_prompt: str) -> str:
    if template_name in PROMPT_TEMPLATES:
        template = PROMPT_TEMPLATES[template_name]
        return f"{base_prompt}, {template['suffix']}"
    return base_prompt