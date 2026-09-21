#!/usr/bin/env bash
# Start the Ethereal Canvas UI.
#
# Delegates to launch_ethereal_canvas.py, the single source of truth for
# startup: it fixes up sys.path, verifies dependencies, checks the port (with
# a fallback range) and calls modules.ui_gradio.launch_ui with its real
# signature (server_name, server_port, share).
#
# Usage:
#   ./scripts/run.sh                 # port from $PORT, config/server_config.yaml, else 7860
#   PORT=7999 ./scripts/run.sh
#   EC_PYTHON=/path/to/python ./scripts/run.sh   # use a specific interpreter
set -eu pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"

# Bind the interpreter by path rather than relying on `source .venv/bin/activate`
# plus a bare `python`. .venv here was created at a different path (see
# .venv/pyvenv.cfg), so its activate script prepends a .venv/bin entry that no
# longer exists and `python` silently resolves to the ambient system
# interpreter - which carries an incompatible torch/flash-attn pair and dies
# inside `import diffusers`. Resolving the path here keeps the app on .venv and
# makes a missing environment a loud error instead of a wrong interpreter.
PYTHON="${EC_PYTHON:-$APP_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "No usable interpreter at: $PYTHON" >&2
    echo "Create it with scripts/install.sh, or point EC_PYTHON at one that has requirements.txt installed." >&2
    exit 1
fi

# PORT and the rest of the environment are inherited by the launcher.
exec "$PYTHON" launch_ethereal_canvas.py "$@"
