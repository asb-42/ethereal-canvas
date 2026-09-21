"""
Backend adapter interface for unified model operations.
Provides simple routing between different model types.
"""

from .text_to_image import TextToImageBackend
from .image_edit import ImageEditBackend
from typing import Optional, Dict, Any
from .image_inpaint import ImageInpaintBackend


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
        # Prefer an explicit inpaint model; otherwise reuse the edit checkpoint, which
        # carries the same Qwen edit weights.
        self.inpaint_model = config.get('inpaint_model') or config.get('edit_model', 'Qwen/Qwen-Image-Edit-2511')
        
        # Lazy loading backends - only create when needed
        self.backends: Dict[str, Any] = {
            't2i': None,
            'edit': None,
            'inpaint': None,
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
    
    def _get_inpaint_backend(self):
        """Lazy load the Inpaint backend.

        Loaded only on the first inpaint request, so re-enabling this route costs
        no second model at startup - which is why it had been disabled.
        """
        if self.backends['inpaint'] is None:
            try:
                print("🔧 Loading Inpaint backend on demand...")
                self.backends['inpaint'] = ImageInpaintBackend(self.inpaint_model)
                self.backends['inpaint'].load()
                print("✓ Inpaint backend loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Inpaint backend: {e}")
                raise
        return self.backends['inpaint']
    
    def inpaint(self, image, mask, prompt):

        """Route image inpainting to the inpaint backend.

        

        Deliberately does not fall back to the edit backend: the edit pipelines

        cannot take a mask, so that fallback silently dropped the user's mask and

        returned an edit of the whole picture. The backend now raises instead.

        """

        return self._get_inpaint_backend().inpaint(image, mask, prompt)

    
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
            return self._get_inpaint_backend()
        return None
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            't2i_model': self.t2i_model,
            'edit_model': self.edit_model,
            'inpaint_model': self.inpaint_model,
            'inpaint_loaded': self.backends.get('inpaint') is not None,
            'lazy_loading': True
        }
