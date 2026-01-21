# Ethereal Canvas - User Guide

## Overview
Ethereal Canvas is an AI-powered image generation and editing application using Qwen models.

## Features

### 🖼️ Generate (Text-to-Image)
- **Model**: Qwen-Image-2512
- **Purpose**: Create images from text descriptions
- **Input**: Text prompt + optional seed
- **Output**: Generated image file

### ✏️ Edit (Image-to-Image)  
- **Model**: Qwen-Image-Edit-2511
- **Purpose**: Edit existing images with text descriptions
- **Input**: Image + edit prompt + optional seed
- **Output**: Edited image file

## Quick Start

### 1. Installation
```bash
# For Pinokio (recommended)
bash scripts/install_pinokio.sh

# For developers
bash scripts/install.sh
```

### 2. Launch Application
```bash
# Start the web interface
python run_ethereal_canvas.py
```

The UI will open in your browser at `http://localhost:7860`

### 3. Generate Images
1. Go to the **Generate** tab
2. Enter a text description of the image you want
3. (Optional) Set a seed for reproducible results
4. Click **Generate Image**
5. Wait for processing and download the result

### 4. Edit Images
1. Go to the **Edit** tab
2. Upload an image you want to edit
3. Enter a text description of the changes you want
4. (Optional) Set a seed for reproducible results
5. Click **Edit Image**
6. Wait for processing and download the result

## Advanced Usage

### Seeds
- Seeds control the randomness of generation
- Same seed + same prompt = same result
- Leave empty for random generation
- Use integers (e.g., 42, 12345)

### Prompts
- Be descriptive and specific
- Include style information (e.g., "photorealistic", "oil painting")
- Use natural language descriptions
- Examples:
  - "A serene mountain landscape at sunset, photorealistic"
  - "Convert this photo to anime style, vibrant colors"

### File Formats
- **Input**: PNG, JPG, JPEG (max 50MB)
- **Output**: PNG format
- **Location**: `outputs/` directory

## Troubleshooting

### Installation Issues
- **Python version**: Requires Python 3.10 or higher
- **Dependencies**: Run `pip install -r requirements.txt` if errors occur
- **CUDA**: Optional - CPU will work but slower

### Generation Issues
- **Slow processing**: Normal for AI models, especially on CPU
- **Poor quality**: Try different prompts or adjust descriptions
- **Errors**: Check logs in the UI status area

### Performance Tips
- **GPU**: CUDA acceleration greatly improves speed
- **Memory**: Close other applications for better performance
- **Quality**: Higher quality takes longer to generate

## Model Information

### Qwen-Image-2512
- **Type**: Text-to-Image Generation
- **Capabilities**: Photorealistic and artistic image generation
- **Size**: ~2-3GB download
- **Cache Location**: `models/Qwen-Image-2512/`

### Qwen-Image-Edit-2511
- **Type**: Image-to-Image Editing
- **Capabilities**: Image modification and enhancement
- **Size**: ~2-3GB download
- **Cache Location**: `models/Qwen-Image-Edit-2511/`

## Configuration

### Server Settings
Edit `config/ui_config.yaml` to change:
- Server port (default: 7860)
- Host address (default: 0.0.0.0)
- Share links (default: false)

### Model Settings
Edit `config/model_config.yaml` to change:
- Model names/versions
- Backend settings
- Precision settings

## Support

For issues and support:
1. Check the status log in the UI
2. Review console output
3. Consult this guide
4. Check the GitHub repository

---

**Enjoy creating with Ethereal Canvas!** 🎨✨