"""
Markdown-based logging and Git integration.
"""

import subprocess
from datetime import datetime
from pathlib import Path


LOG_PATH = Path("logs/runlog.md")


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