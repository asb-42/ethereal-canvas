"""
Model registry for Ethereal Canvas.

Provides dynamic backend selection and mode management.
"""

from typing import Dict, Any, Optional, Type, List
from pathlib import Path
import importlib

from .base import GenerationBackend, InpaintingBackend

# Model registry
MODEL_REGISTRY: Dict[str, Type[GenerationBackend]] = {}

def register_backend(model_id: str, backend_class: Type[GenerationBackend]) -> None:
    """Register a backend class for a model ID."""
    MODEL_REGISTRY[model_id] = backend_class

def get_backend_class(model_id: str) -> Optional[Type[GenerationBackend]]:
    """Get backend class for a model ID."""
    return MODEL_REGISTRY.get(model_id)

def get_available_backends() -> Dict[str, Type[GenerationBackend]]:
    """Get all registered backends."""
    return MODEL_REGISTRY.copy()

def create_backend(model_id: str, model_config: Dict[str, Any]) -> GenerationBackend:
    """Create a backend instance from model ID and configuration."""
    backend_class = get_backend_class(model_id)
    if not backend_class:
        raise ValueError(f"No backend registered for model: {model_id}")
    
    # Extract backend-specific configuration
    backend_config = {}
    for key, value in model_config.items():
        if key.startswith('backend_'):
            backend_config[key[8:]] = value
    
    return backend_class(model_id, **backend_config)

# Lazy registration function - call only when needed
def register_all_backends() -> None:
    """Register all available backend classes."""
    try:
        # Import and register all backend modules
        from ..backends import text_to_image, image_edit, image_inpaint
        
        register_backend("text-to-image", text_to_image.TextToImageBackend)
        register_backend("image-edit", image_edit.ImageEditBackend)
        register_backend("image-inpaint", image_inpaint.ImageInpaintBackend)
        print("✅ All backends registered successfully")
    except ImportError as e:
        print(f"Warning: Failed to register backends: {e}")