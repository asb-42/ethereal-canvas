"""
Mask image loader for inpainting.
Masks are expected to be single-channel images.
"""

from PIL import Image
from modules.img_read.reader import ImageData

def read_mask(path: str) -> ImageData:
    img = Image.open(path).convert("L")
    return ImageData(
        pixels=img,
        width=img.width,
        height=img.height,
        format="L",
        metadata={}
    )