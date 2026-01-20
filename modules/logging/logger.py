"""
Markdown-based logging and Git integration.
"""

import subprocess
import sys
import torch
from datetime import datetime
from pathlib import Path


LOG_PATH = Path("logs/runlog.md")


def log_session_header():
    LOG_PATH.parent.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except:
        git_hash = "unknown"
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    
    header = f"# Session Started: {timestamp}\n"
    header += f"- Git commit: {git_hash}\n"
    header += f"- Python: {python_version}\n"
    header += f"- CUDA available: {cuda_available}\n"
    header += f"- GPU: {gpu_name}\n\n"
    
    with open(LOG_PATH, "a") as f:
        f.write(header)
    
    subprocess.run(["git", "add", str(LOG_PATH)])
    subprocess.run(["git", "commit", "-m", "Log: Session started"])


def log_step(message: str, task_type=None, prompt=None, seed=None, model=None, input_path=None, output_path=None, status=None):
    LOG_PATH.parent.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().isoformat()

    entry = f"## {timestamp}\n"
    if task_type:
        entry += f"- Task type: {task_type}\n"
    if prompt:
        entry += f"- Prompt: {prompt}\n"
    if seed is not None:
        entry += f"- Seed: {seed}\n"
    if model:
        entry += f"- Model: {model}\n"
    if input_path:
        entry += f"- Input: {input_path}\n"
    if output_path:
        entry += f"- Output: {output_path}\n"
    if status:
        entry += f"- Status: {status}\n"
    entry += f"\n{message}\n\n"

    with open(LOG_PATH, "a") as f:
        f.write(entry)

    subprocess.run(["git", "add", str(LOG_PATH)])
    subprocess.run(["git", "commit", "-m", f"Log: {message[:60]}"])