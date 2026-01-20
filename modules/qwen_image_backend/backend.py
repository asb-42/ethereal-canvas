"""
Qwen-Image backend implementation using HuggingFace transformers.
"""

import torch
from modules.model_adapter.adapter import ModelAdapter
from modules.img_read.reader import ImageData
from modules.prompt_engine.engine import PromptObject


class QwenImageBackend(ModelAdapter):

    def __init__(self, model_config: dict):
        self.model_config = model_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def load(self):
        # Model loading will be implemented in Phase III
        pass

    def generate_image(self, prompt: PromptObject) -> ImageData:
        raise NotImplementedError("Generation not yet implemented")

    def edit_image(self, image: ImageData, prompt: PromptObject) -> ImageData:
        raise NotImplementedError("Editing not yet implemented")

    def shutdown(self):
        self.model = None