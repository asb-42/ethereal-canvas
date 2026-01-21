# User Guide

## Installation

### End User (Pinokio One-Click Installer)

Ethereal Canvas is designed to be installed **directly from the Pinokio platform**. There is no need to manually clone or configure the repository.

1. Open the **Pinokio application** on your system.
2. Navigate to the **Pinokio Apps catalog**.
3. Locate **"Ethereal Canvas"** and click **"Install"** or **"One-Click Installer"** button.
4. Wait while Pinokio downloads the app, sets up all dependencies, and configures the environment automatically.
5. Once installation completes, **launch Ethereal Canvas directly from within Pinokio**.

> **Note:** The One-Click Installer handles all system setup, including Python environments, required libraries, and runtime configuration.

### Developer Setup (Optional)

If you intend to **inspect, modify, or contribute** to the codebase, you can set up a local development environment.

1. Clone the repository from GitHub:

```bash
git clone https://github.com/asb-42/ethereal-canvas.git
```

2. Enter the repository directory:

```bash
cd ethereal-canvas
```

3. Set up a Python virtual environment (venv or conda):

```bash
# Example using venv
python3 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

4. Launch the application locally for testing:

```bash
python run_ethereal_canvas.py
```

**Important:** This workflow is intended for developers only. End users should always use the Pinokio One-Click Installer.

## Launching the App

The Gradio UI will be available on the configured port (default: 7860). You can access it from your browser.

## Text → Image

1. Enter a descriptive prompt in the text box
2. Optionally set a seed for reproducible results
3. Click "Generate"
4. The generated image will appear in the output panel

## Image → Image

1. Upload an image using the file input
2. Enter your editing prompt
3. Optionally set a seed
4. Click "Edit"
5. The edited image will be shown

## Seeds & Reproducibility

- Seeds control the random generation process
- Same prompt + same seed = identical output
- Leave seed empty for random generation
- Seeds are logged for traceability

## Output Metadata

Every generated image includes embedded metadata:
- Original prompt
- Seed used
- Model version
- Generation parameters

## Logs & Audit Trail

All operations are logged in `logs/runlog.md`:
- Structured markdown format
- Git-committed after each operation
- Includes system fingerprint
- Complete audit trail for reproducibility

## Troubleshooting

- **Port already in use**: Change PORT environment variable or stop conflicting services
- **CUDA out of memory**: Restart the application to clear GPU memory
- **Model loading fails**: Check internet connection and run install script again