"""
Job runner orchestration for tasks.
Enqueues generate/edit tasks and logs.
"""

from modules.prompt_engine.engine import normalize_prompt
from modules.prompt_engine.seed import set_global_seed
from modules.img_read.reader import read_image
from modules.img_write.writer import write_image
from modules.logging.logger import log_step, log_session_header
from modules.qwen_image_backend.loader import QwenImageBackend
from modules.qwen_image_backend.memory import clear_gpu
import yaml

# Initialize logging on first import
log_session_header()

with open("config/model_config.yaml") as f:
    MODEL_CONFIG = yaml.safe_load(f)

backend = QwenImageBackend(MODEL_CONFIG)
backend.load()

SUPPORTED_TASKS = ["generate", "edit", "inpaint"]

def execute_task(task_type, prompt_text, seed=None, input_path=None, mask_path=None):
    # Validation
    if task_type == "inpaint" and not mask_path:
        raise ValueError("Inpainting requires a mask image")
    
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    prompt_obj = normalize_prompt(prompt_text, seed)
    set_global_seed(prompt_obj.seed)
    
    log_step(
        f"Task {task_type}: prompt='{prompt_obj.prompt_text}', seed={prompt_obj.seed}",
        task_type=task_type,
        prompt=prompt_obj.prompt_text,
        seed=prompt_obj.seed,
        model=getattr(backend, 'model_id', 'unknown'),
        input_path=input_path,
        status="started"
    )

    # Input
    image_data = None
    if input_path:
        image_data = read_image(input_path)

    # Choose
    try:
        if task_type == "generate":
            result = backend.generate_image(prompt_obj)
        elif task_type == "edit":
            if not image_data:
                raise ValueError("Edit task requires an input image")
            result = backend.edit_image(image_data, prompt_obj)
        elif task_type == "inpaint":
            from modules.img_read.mask_reader import read_mask
            if not image_data or not mask_path:
                raise ValueError("Inpainting requires both image and mask")
            mask_data = read_mask(mask_path)
            result = backend.inpaint(image_data, mask_data, prompt_obj)
        else:
            raise ValueError("Unknown task type")

        # Write
        out_path = f"outputs/{task_type}_{prompt_obj.seed}.png"
        write_image(result, out_path, result.metadata)
        
        log_step(
            f"Output saved to {out_path}",
            task_type=task_type,
            output_path=out_path,
            status="completed"
        )
        
        clear_gpu()
        return out_path
        
    except Exception as e:
        log_step(
            f"Task failed: {str(e)}",
            task_type=task_type,
            status="failed"
        )
        raise