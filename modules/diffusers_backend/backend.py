"""
Diffusers-based image backend.
Used for models that provide native inpainting and image editing.
"""

import torch
import io
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, QwenImageEditPlusPipeline
from modules.model_adapter.adapter import ModelAdapter
from modules.prompt_engine.engine import PromptObject
from modules.img_read.reader import ImageData
from PIL import Image

class DiffusersBackend(ModelAdapter):

    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.edit_pipeline = None
        self.inpaint_pipeline = None
        self.model_id = None

    def load(self):
        model_name = self.config["model_name"]
        
        # Load text-to-image pipeline
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(self.device)
        
        # Load edit-specific pipeline if available
        try:
            self.edit_pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name).to(self.device)
        except ImportError:
            print(f"QwenImageEditPlusPipeline not available, using base pipeline for editing")
            self.edit_pipeline = self.pipeline
            
        # Load inpainting pipeline
        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_name).to(self.device)
        
        self.model_id = model_name

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
        # Use dedicated edit pipeline if available, fallback to generate_image
        enhanced_prompt = f"{prompt.prompt_text} (edited image)"
        
        if self.edit_pipeline is not None:
            # Try Qwen-specific edit pipeline
            inputs = self.edit_pipeline.tokenizer(
                enhanced_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            # Include the input image as conditioning
            # Note: This is a simplified approach - real editing may require different pipeline usage
            outputs = self.edit_pipeline.generate(**inputs)
            image_bytes = outputs[0]
        else:
            # Fallback to regular generation
            image_bytes = self.edit_pipeline(enhanced_prompt).images[0]
        
        img = Image.open(io.BytesIO(image_bytes))
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

    def inpaint(self, image: ImageData, mask: ImageData, prompt: PromptObject) -> ImageData:
        result_image = self.inpaint_pipeline(
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
        del self.edit_pipeline
        del self.inpaint_pipeline
        self.pipeline = None
        self.edit_pipeline = None
        self.inpaint_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()