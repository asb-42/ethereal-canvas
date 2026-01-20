"""
Markdown-based logging and Git integration.
"""

import subprocess
from datetime import datetime
from pathlib import Path


LOG_PATH = Path("logs/runlog.md")


def log_step(message: str):
    LOG_PATH.parent.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().isoformat()

    entry = f"## {timestamp}\n{message}\n\n"
    with open(LOG_PATH, "a") as f:
        f.write(entry)

    subprocess.run(["git", "add", str(LOG_PATH)])
    subprocess.run(["git", "commit", "-m", f"Log: {message[:60]}"])