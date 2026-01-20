"""
Diffusers-based image backend.
Used for models that provide native inpainting and image editing.
"""

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from modules.model_adapter.adapter import ModelAdapter
from modules.prompt_engine.engine import PromptObject
from modules.img_read.reader import ImageData
from PIL import Image

class DiffusersBackend(ModelAdapter):

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.model_id = None

    def load(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.config["model_name"]
        ).to(self.device)
        self.model_id = self.config["model_name"]

    def generate_image(self, prompt: PromptObject) -> ImageData:
        image = self.pipeline(prompt.prompt_text).images[0]
        metadata = {
            "prompt": prompt.prompt_text,
            "seed": prompt.seed,
            "params": str(prompt.params),
            "model": self.model_id
        }
        return ImageData(
            pixels=image,
            width=image.width,
            height=image.height,
            format=image.format or "PNG",
            metadata=metadata
        )

    def edit_image(self, image: ImageData, prompt: PromptObject) -> ImageData:
        return self.generate_image(prompt)

    def inpaint(self, image: ImageData, mask: ImageData, prompt: PromptObject) -> ImageData:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.config["model_name"]
        ).to(self.device)
        
        result_image = pipe(
            prompt=prompt.prompt_text,
            image=image.pixels,
            mask_image=mask.pixels
        ).images[0]
        
        metadata = {
            "prompt": prompt.prompt_text,
            "seed": prompt.seed,
            "params": str(prompt.params),
            "model": self.model_id
        }
        return ImageData(
            pixels=result_image,
            width=result_image.width,
            height=result_image.height,
            format=result_image.format or "PNG",
            metadata=metadata
        )

    def shutdown(self):
        del self.pipeline
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()