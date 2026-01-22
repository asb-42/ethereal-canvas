#!/usr/bin/env python3
"""
Simple test of launcher functionality.
"""

import os
import sys
from pathlib import Path

# Test virtual environment detection
venv_python = None
if os.path.exists(".venv/bin/python"):
    venv_python = ".venv/bin/python"
elif os.path.exists("venv/bin/python"):
    venv_python = "venv/bin/python"

print(f"🔍 Virtual environment detected: {venv_python}")
print(f"🐍 Current Python: {sys.executable}")

# Test module imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from modules.ui_gradio.simple_ui import launch_ui
    print("✅ UI module import successful")
    
    # Test configuration loading
    import yaml
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)
    print(f"✅ Configuration loaded: {list(config.keys())}")
    
    print("🎉 Launcher test successful!")
    print("🚀 Ready to launch UI...")
    
except Exception as e:
    print(f"❌ Launcher test failed: {e}")
    import traceback
    traceback.print_exc()