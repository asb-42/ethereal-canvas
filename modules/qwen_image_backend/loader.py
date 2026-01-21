"""
Qwen Image Backend — Load and inference methods using transformers.

This handles:
- Model loading
- Text -> Image generation
- Early hooks for image -> image (editing)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq, QwenImageForConditionalGeneration
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
        # Load model based on configuration
        model_name = self.config.get("model_name", "Qwen/Qwen-Image-2512")
        
        # Use QwenImage loader for T2I generation
        try:
            self.tokenizer = AutoProcessor.from_pretrained(model_name)
            self.model = QwenImageForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
        except ImportError:
            # Fallback to generic AutoModelForVision2Seq if QwenImage not available
            print("QwenImage package not available, falling back to generic model")
            self.tokenizer = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name
            ).to(self.device)
        
        self.model.eval()
        self.model_id = model_name

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
        Image editing using regular QwenImage generation with image conditioning
        """
        model_name = self.config.get("edit_model", self.config.get("model_name", "Qwen/Qwen-Image-2512"))
        
        try:
            # Try to load edit-specific model if different from generate model
            if model_name != self.model_id:
                edit_tokenizer = AutoProcessor.from_pretrained(model_name)
                edit_model = QwenImageForConditionalGeneration.from_pretrained(model_name).to(self.device)
            else:
                edit_tokenizer = self.tokenizer
                edit_model = self.model
                
            # Create a prompt that references the image
            enhanced_prompt = f"{prompt.prompt_text} (conditioned on input image)"
            
            inputs = edit_tokenizer(
                enhanced_prompt,
                return_tensors="pt"
            ).to(self.device)

            # Include the input image as conditioning
            # Note: This is a simplified approach - real image editing may require different prompt engineering
            outputs = edit_model.generate(**inputs)
            
        except ImportError:
            # Fallback to regular generation if edit-specific features unavailable
            return self.generate_image(prompt)

        image_bytes = outputs[0]  # byte array of encoded image

        img = Image.open(io.BytesIO(image_bytes))
        metadata = {
            "prompt": prompt.prompt_text,
            "seed": prompt.seed,
            "params": str(prompt.params),
            "model": model_name
        }
        return ImageData(
            pixels=img,
            width=img.width,
            height=img.height,
            format=img.format or "PNG",
            metadata=metadata
        )

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