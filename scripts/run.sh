#!/usr/bin/env bash
source .venv/bin/activate

PORT=${PORT:-7860}
if lsof -i:$PORT &> /dev/null; then
  echo "Port $PORT already in use"
  exit 1
fi

python - << 'PYCODE'
from modules.ui_gradio.interface import launch_ui
import yaml

cfg = yaml.safe_load(open("config/server_config.yaml"))
launch_ui(None, None, None, cfg)
PYCODE