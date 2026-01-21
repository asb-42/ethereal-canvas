"""
Backend adapter interface for unified model operations.
Provides simple routing between different model types.
"""

from .text_to_image import TextToImageBackend
from .image_edit import ImageEditBackend
from .image_inpaint import ImageInpaintBackend


class BackendAdapter:
    """
    Simplified adapter that routes operations to appropriate backends.
    """
    
    def __init__(self, config: dict):
        """Initialize adapter with backend routing."""
        self.config = config
        # Extract model names from config or use defaults
        t2i_model = config.get('generate_model', 'Qwen/Qwen-Image-2512')
        edit_model = config.get('edit_model', 'Qwen/Qwen-Image-Edit-2511')
        inpaint_model = config.get('edit_model', 'Qwen/Qwen-Image-Edit-2511')  # Same model for inpainting
        
        self.t2i_backend = TextToImageBackend(t2i_model)
        self.edit_backend = ImageEditBackend(edit_model)
        self.inpaint_backend = ImageInpaintBackend(inpaint_model)
    
    def generate(self, prompt):
        """Route text-to-image generation."""
        return self.t2i_backend.generate(prompt)
    
    def edit(self, prompt, input_path):
        """Route image editing."""
        return self.edit_backend.edit(prompt, input_path)
    
    def inpaint(self, image, mask, prompt):
        """Route image inpainting."""
        return self.inpaint_backend.inpaint(image, mask, prompt)
    
    def load(self):
        """Load all backends."""
        for backend in [self.t2i_backend, self.edit_backend, self.inpaint_backend]:
            if hasattr(backend, 'load'):
                backend.load()
    
    def shutdown(self):
        """Shutdown all backends."""
        for backend in [self.t2i_backend, self.edit_backend, self.inpaint_backend]:
            if hasattr(backend, 'cleanup'):
                backend.cleanup()
    
    def get_backend(self, task_type: str):
        """Get backend for specific task type."""
        backends = {
            'generate': self.t2i_backend,
            'edit': self.edit_backend,
            'inpaint': self.inpaint_backend
        }
        return backends.get(task_type)
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            't2i_backend': str(self.t2i_backend),
            'edit_backend': str(self.edit_backend),
            'inpaint_backend': str(self.inpaint_backend)
        }