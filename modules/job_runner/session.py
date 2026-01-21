"""
Session persistence.
"""

import json
from pathlib import Path

SESSION_FILE = Path("logs/session.json")

def save_session(data):
    SESSION_FILE.parent.mkdir(exist_ok=True)
    SESSION_FILE.write_text(json.dumps(data, indent=2))

def load_session():
    if SESSION_FILE.exists():
        return json.loads(SESSION_FILE.read_text())
    return {}