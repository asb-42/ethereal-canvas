"""
Qwen Image Backend — Load and inference methods using transformers.

This handles:
- Model loading
- Text -> Image generation
- Early hooks for image -> image (editing)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq
from modules.model_adapter.adapter import ModelAdapter
from modules.prompt_engine.engine import PromptObject
from modules.img_read.reader import ImageData
from PIL import Image
import io

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

class QwenImageBackend(ModelAdapter):

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def load(self):
        # AutoProcessor is used for vision + language
        self.tokenizer = AutoProcessor.from_pretrained("Qwen/Qwen-Image-2512")
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen-Image-2512"
        ).to(self.device)
        self.model.eval()
        self.model_id = self.model.config._name_or_path

    def generate_image(self, prompt: PromptObject) -> ImageData:
        """
        Generate an image from text prompt
        """
        inputs = self.tokenizer(
            prompt.prompt_text,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(**inputs)
        image_bytes = outputs[0]  # byte array of encoded image

        image = Image.open(io.BytesIO(image_bytes))
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
        """
        For now alias to generate_image with context
        """
        return self.generate_image(prompt)

    def inpaint(self, image: ImageData, mask: ImageData, prompt: PromptObject) -> ImageData:
        """
        Native inpainting placeholder.
        """
        raise NotImplementedError(
            "Qwen-Image inpainting not yet exposed via transformers"
        )

    def shutdown(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()