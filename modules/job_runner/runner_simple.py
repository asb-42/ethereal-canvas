"""
Job runner orchestration for Ethereal Canvas.
Simple version that works without heavy dependencies.
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.backends.adapter import BackendAdapter

# Simple logging that doesn't depend on torch
def log_simple(message: str, status="INFO"):
    """Simple logging function."""
    timestamp = os.popen('date +"%Y-%m-%d %H:%M:%S"').read().strip()
    print(f"[{timestamp}] {status}: {message}")

SUPPORTED_TASKS = ["generate", "edit"]

# Global backend adapter
backend_adapter = None

def load_backend():
    """Load backend adapter."""
    global backend_adapter
    try:
        config = {
            'generate_model': 'Qwen/Qwen-Image-2512',
            'edit_model': 'Qwen/Qwen-Image-Edit-2511'
        }
        backend_adapter = BackendAdapter(config)
        backend_adapter.load()
        log_simple("Backend adapter loaded")
    except Exception as e:
        log_simple(f"Failed to load backend: {e}", "ERROR")

def execute_task(task_type, prompt_text=None, seed=None, input_path=None, mask_path=None):
    """Execute a single task."""
    
    if task_type not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {task_type}")
    
    log_simple(f"Executing {task_type} task")
    
    try:
        # Load backend if needed
        if not backend_adapter:
            load_backend()
        
        # Execute task based on type
        if task_type == "generate":
            output_path = backend_adapter.generate(prompt_text)
        elif task_type == "edit":
            if not input_path:
                raise ValueError("edit task requires input_path")
            output_path = backend_adapter.edit(input_path, prompt_text)
        else:
            raise ValueError(f"Unsupported task: {task_type}")
        
        log_simple(f"Task {task_type} completed: {output_path}")
        return output_path
        
    except Exception as e:
        log_simple(f"Task {task_type} failed: {str(e)}", "ERROR")
        raise

def get_model_info():
    """Get model information."""
    if not backend_adapter:
        load_backend()
    return backend_adapter.get_model_info()

def cleanup():
    """Cleanup resources."""
    global backend_adapter
    if backend_adapter:
        try:
            backend_adapter.shutdown()
            log_simple("Backend adapter shutdown")
        except:
            pass
        backend_adapter = None

def test_backend():
    """Test backend functionality."""
    if not backend_adapter:
        load_backend()
    
    try:
        info = backend_adapter.get_model_info()
        print("✅ Backend test successful")
        print(f"Backend info: {info}")
        return True
    except Exception as e:
        print(f"❌ Backend test failed: {e}")
        return False