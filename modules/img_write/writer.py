"""
Image output handling.

Responsibilities:
- Write image to disk
- Embed metadata
- Ensure reproducibility information is stored
"""

from typing import Dict, Optional, Union
from PIL import Image
import os
from modules.img_read.reader import ImageData


def write_image(image_data: Union[ImageData, Image.Image], out_path: str, metadata: Optional[Dict] = None) -> str:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if isinstance(image_data, ImageData):
        img = image_data.pixels.copy()
        if metadata:
            img.info.update(metadata)
    elif isinstance(image_data, Image.Image):
        img = image_data
        if metadata and hasattr(img, 'info'):
            img.info.update(metadata)
    else:
        raise TypeError(f"Expected ImageData or PIL.Image, got {type(image_data)}")

    img.save(out_path)
    return out_path