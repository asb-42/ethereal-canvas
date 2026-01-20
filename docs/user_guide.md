# User Guide

## Installation (Pinokio)

1. Clone the repository
2. Run `scripts/install.sh` to set up the environment and download the model
3. Launch the application with `scripts/run.sh`

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