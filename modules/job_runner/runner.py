"""
Job runner orchestration for tasks.
Simple version using the new backend adapter.
"""

import os
import sys
import yaml
from modules.prompt_engine.engine import normalize_prompt
from modules.prompt_engine.seed import set_global_seed
from modules.img_read.reader import read_image
from modules.img_write.writer import write_image
from modules.logging.logger import log_step, log_session_header
from modules.backends.adapter import BackendAdapter

# Initialize logging
log_session_header()

# Load configuration
try:
    with open("config/model_config.yaml") as f:
        MODEL_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    MODEL_CONFIG = {
        'text_to_image_model': 'default-t2i',
        'image_edit_model': 'default-edit', 
        'image_inpaint_model': 'default-inpaint'
    }

# Initialize backend adapter
backend_adapter = BackendAdapter(MODEL_CONFIG)

# Load backends
backend_adapter.load()

SUPPORTED_TASKS = ["generate", "edit", "inpaint"]

def execute_task(task_type, prompt_text, seed=None, input_path=None, mask_path=None):
    """Execute a single task."""
    
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task_type}")
    
    # Set seed if provided
    if seed is not None:
        set_global_seed(seed)
    
    # Normalize prompt
    prompt = normalize_prompt(prompt_text)
    
    output_path = None
    try:
        if task_type == "generate":
            output_path = backend_adapter.generate(prompt)
            
        elif task_type == "edit":
            if not input_path:
                raise ValueError("edit task requires input_path")
            output_path = backend_adapter.edit(prompt, input_path)
            
        elif task_type == "inpaint":
            if not input_path or not mask_path:
                raise ValueError("inpaint task requires input_path and mask_path")
            image = read_image(input_path)
            mask = read_image(mask_path)
            output_path = backend_adapter.inpaint(image, mask, prompt)
        
        # Log successful completion
        log_step(
            message=f"Task {task_type} completed successfully",
            task_type=task_type,
            prompt=prompt_text,
            input_path=input_path,
            output_path=output_path,
            status="success"
        )
        
        return output_path
        
    except Exception as e:
        # Log failure
        log_step(
            message=f"Task {task_type} failed: {str(e)}",
            task_type=task_type,
            prompt=prompt_text,
            input_path=input_path,
            output_path=None,
            status="failed"
        )
        raise

def cleanup():
    """Cleanup resources."""
    if backend_adapter:
        backend_adapter.shutdown()

def get_model_info():
    """Get model information."""
    return backend_adapter.get_model_info()

def test_backend():
    """Test backend functionality."""
    print("Testing backend...")
    try:
        # Test generate
        result = execute_task("generate", "test image")
        print(f"Generate test: {result}")
        
        # Test model info
        info = get_model_info()
        print(f"Model info: {info}")
        
        return True
    except Exception as e:
        print(f"Backend test failed: {e}")
        return False