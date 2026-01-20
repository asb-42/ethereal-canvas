"""
Image input handling.

Responsibilities:
- Validate image file path
- Load JPEG or PNG
- Extract basic metadata
- Convert into internal ImageData structure
"""

from dataclasses import dataclass
from typing import Dict
from PIL import Image
import os


@dataclass
class ImageData:
    pixels: Image.Image
    width: int
    height: int
    format: str
    metadata: Dict


def read_image(input_path: str) -> ImageData:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    img = Image.open(input_path)
    metadata = dict(img.info)

    return ImageData(
        pixels=img,
        width=img.width,
        height=img.height,
        format=img.format or "unknown",
        metadata=metadata
    )