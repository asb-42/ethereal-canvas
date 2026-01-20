## Ethereal Canvas 0.1.0-alpha

### Features
- Text to Image generation via Qwen-Image-2512
- Image to Image editing capabilities
- Deterministic generation with seed control
- Metadata embedding in output images
- Network-accessible Gradio UI
- Git-backed audit trail with Markdown logs
- Pinokio one-click installer support

### Technical Highlights
- Modular Unix-style architecture
- Extensible model adapter interface
- GPU memory management with automatic cleanup
- Global seed control for reproducibility
- Comprehensive error handling and logging
- System fingerprinting for audit trails

### Known Limitations
- No native inpainting yet (interface defined, placeholder implemented)
- Single-user execution model
- transformers backend only (Qwen-Image-2512)
- No batch generation support
- Requires significant GPU memory for large models

### Installation
1. Clone repository
2. Run `scripts/install.sh` for automated setup
3. Launch with `scripts/run.sh` or via Pinokio

### Quick Start
- Access UI at http://localhost:7860
- Enter prompt, optional seed, click Generate
- For editing, upload image first, then click Edit
- All outputs saved to `outputs/` directory
- Full audit trail available in `logs/runlog.md`

### System Requirements
- Python 3.10+
- CUDA GPU (recommended) or CPU (slower)
- 8GB+ RAM (16GB+ recommended for GPU)
- 10GB+ disk space for model cache