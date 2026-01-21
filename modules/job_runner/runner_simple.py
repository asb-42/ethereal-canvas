"""
Job runner orchestration for tasks.
Simple version that works without torch dependencies.
"""

import os
import sys
import yaml
from modules.prompt_engine.engine import normalize_prompt
from modules.backends.adapter import BackendAdapter

# Simple logging that doesn't depend on torch
def log_simple(message: str, status="INFO"):
    """Simple logging function."""
    timestamp = os.popen('date +"%Y-%m-%d %H:%M:%S"').read().strip()
    print(f"[{timestamp}] {status}: {message}")

# Load configuration
try:
    with open("config/model_config.yaml") as f:
        MODEL_CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    MODEL_CONFIG = {
        'generate_model': 'Qwen/Qwen-Image-2512',
        'edit_model': 'Qwen/Qwen-Image-Edit-2511',
        'backend': 'diffusers'
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
    
    log_simple(f"Executing {task_type} task")
    
    try:
        if task_type == "generate":
            output_path = backend_adapter.generate(prompt_text)
            
        elif task_type == "edit":
            if not input_path:
                raise ValueError("edit task requires input_path")
            output_path = backend_adapter.edit(prompt_text, input_path)
            
        elif task_type == "inpaint":
            if not input_path or not mask_path:
                raise ValueError("inpaint task requires input_path and mask_path")
            output_path = backend_adapter.inpaint(input_path, mask_path, prompt_text)
        
        log_simple(f"Task {task_type} completed: {output_path}")
        return output_path
        
    except Exception as e:
        log_simple(f"Task {task_type} failed: {str(e)}", "ERROR")
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