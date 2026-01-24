"""
Backend adapter interface for unified model operations.
Provides simple routing between different model types.
"""

from .text_to_image import TextToImageBackend
from .image_edit import ImageEditBackend
from typing import Optional, Dict, Any
# from .image_inpaint import ImageInpaintBackend  # Temporarily disabled


class BackendAdapter:
    """
    Simplified adapter that routes operations to appropriate backends with lazy loading.
    """
    
    def __init__(self, config: dict):
        """Initialize adapter with backend routing (lazy loading)."""
        self.config = config
        # Extract model names from config or use defaults
        self.t2i_model = config.get('generate_model', 'Qwen/Qwen-Image-2512')
        self.edit_model = config.get('edit_model', 'Qwen/Qwen-Image-Edit-2511')
        # self.inpaint_model = config.get('edit_model', 'Qwen/Qwen-Image-Edit-2511')  # Temporarily disabled
        
        # Lazy loading backends - only create when needed
        self.backends: Dict[str, Any] = {
            't2i': None,
            'edit': None,
            # 'inpaint': None,  # Temporarily disabled - same as edit
        }
    
    def _get_t2i_backend(self):
        """Lazy load T2I backend."""
        if self.backends['t2i'] is None:
            try:
                print("🔧 Loading T2I backend on demand...")
                self.backends['t2i'] = TextToImageBackend(self.t2i_model)
                self.backends['t2i'].load()
                print("✅ T2I backend loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load T2I backend: {e}")
                raise
        return self.backends['t2i']
    
    def generate(self, prompt):
        """Route text-to-image generation."""
        return self._get_t2i_backend().generate(prompt)
    
    def _get_edit_backend(self):
        """Lazy load edit backend."""
        if self.backends['edit'] is None:
            try:
                print("🔧 Loading Edit backend on demand...")
                self.backends['edit'] = ImageEditBackend(self.edit_model)
                self.backends['edit'].load()
                print("✅ Edit backend loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Edit backend: {e}")
                raise
        return self.backends['edit']
    
    def edit(self, prompt, input_path):
        """Route image editing."""
        return self._get_edit_backend().edit(prompt, input_path)
    
    def inpaint(self, image, mask, prompt):
        """Route image inpainting (temporarily disabled - using edit backend)."""
        # Temporarily disabled to avoid duplicate model loading
        print("🚫 Inpainting temporarily disabled - using edit backend instead")
        return self._get_edit_backend().edit(prompt, image)
    
    def load(self):
        """Lazy loading - no immediate loading, backends load on demand."""
        print("🔧 Lazy loading enabled - backends will load when first used")
    
    def shutdown(self):
        """Shutdown all loaded backends."""
        for backend_name, backend in self.backends.items():
            if backend is not None and hasattr(backend, 'cleanup'):
                print(f"🔧 Cleaning up {backend_name} backend...")
                backend.cleanup()
    
    def get_backend(self, task_type: str):
        """Get backend for specific task type (lazy loading)."""
        if task_type == 'generate':
            return self._get_t2i_backend()
        elif task_type == 'edit':
            return self._get_edit_backend()
        elif task_type == 'inpaint':
            return self._get_edit_backend()  # Using edit backend for now
        return None
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            't2i_model': self.t2i_model,
            'edit_model': self.edit_model,
            'inpaint_model': 'disabled (using edit backend)',
            'lazy_loading': True
        }