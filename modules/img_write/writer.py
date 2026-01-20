"""
Image output handling.

Responsibilities:
- Write image to disk
- Embed metadata
- Ensure reproducibility information is stored
"""

from typing import Dict
from PIL import Image
import os
from modules.img_read.reader import ImageData


def write_image(image_data: ImageData, out_path: str, metadata: Dict) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img: Image.Image = image_data.pixels.copy()
    img.info.update(metadata)

    img.save(out_path)
    return out_path