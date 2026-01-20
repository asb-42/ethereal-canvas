"""
Abstract model adapter interface.
All image models must implement this API.
"""

from abc import ABC, abstractmethod
from modules.img_read.reader import ImageData
from modules.prompt_engine.engine import PromptObject


class ModelAdapter(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def generate_image(self, prompt: PromptObject) -> ImageData:
        pass

    @abstractmethod
    def edit_image(self, image: ImageData, prompt: PromptObject) -> ImageData:
        pass

    @abstractmethod
    def inpaint(self, image: ImageData, mask: ImageData, prompt: PromptObject) -> ImageData:
        pass

    @abstractmethod
    def shutdown(self):
        pass