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
    
    def _t2i_backend(self, model_name: str):
        """Pick the class that can actually serve this model name."""
        from modules.runtime.paths import is_qwen_image_21
        if is_qwen_image_21(model_name):
            from .image_21 import QwenImage21Backend
            return QwenImage21Backend(model_name)
        return TextToImageBackend(model_name)

    def _edit_backend(self, model_name: str):
        """Pick the class that can actually serve this model name."""
        from modules.runtime.paths import is_qwen_image_21
        if is_qwen_image_21(model_name):
            from .image_21 import QwenImage21Backend
            return QwenImage21Backend(model_name)
        return ImageEditBackend(model_name)

    def _shared_21_instance(self):
        """An already-loaded unified backend serving the same checkpoint.

        The 2.1 release is one checkpoint for both roles, so when both roles
        name it there is no reason to hold it twice: a second load costs
        another ~4 minutes and another ~31 GiB of residency. Returns the
        shared QwenImage21Backend, or None when the roles name different
        checkpoints (legacy pair) or nothing suitable is loaded yet.
        """
        for key in ("t2i", "edit"):
            backend = self.backends.get(key)
            if (backend is not None and hasattr(backend, "generate_image")
                    and self.t2i_model == self.edit_model
                    and backend.model_name == self.t2i_model):
                return backend
        return None

    def _get_t2i_backend(self):
        """Lazy load T2I backend."""
        if self.backends['t2i'] is None:
            shared = self._shared_21_instance()
            if shared is not None:
                self.backends['t2i'] = shared
                return shared
            try:
                print("🔧 Loading T2I backend on demand...")
                self.backends['t2i'] = self._t2i_backend(self.t2i_model)
                self.backends['t2i'].load()
                print("✅ T2I backend loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load T2I backend: {e}")
                raise
        return self.backends['t2i']
    
    def generate(self, prompt, **kwargs):
        """Route text-to-image generation."""
        backend = self._get_t2i_backend()
        if kwargs and not hasattr(backend, "generate_image"):
            # Legacy backends speak generate(prompt) only; extras like the
            # text-encoder choice are 2.1-only and must not break them.
            kwargs = {}
        return backend.generate(prompt, **kwargs)
    
    def _get_edit_backend(self):
        """Lazy load edit backend."""
        if self.backends['edit'] is None:
            shared = self._shared_21_instance()
            if shared is not None:
                self.backends['edit'] = shared
                return shared
            try:
                print("🔧 Loading Edit backend on demand...")
                self.backends['edit'] = self._edit_backend(self.edit_model)
                self.backends['edit'].load()
                print("✅ Edit backend loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load Edit backend: {e}")
                raise
        return self.backends['edit']
    
    def edit(self, prompt, input_path, **kwargs):
        """Route image editing."""
        backend = self._get_edit_backend()
        if kwargs and not hasattr(backend, "generate_image"):
            kwargs = {}
        return backend.edit(prompt, input_path, **kwargs)
    
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
            'shared_21_pipeline': (
                self.backends.get('t2i') is not None
                and self.backends.get('t2i') is self.backends.get('edit')),
            'lazy_loading': True
        }
