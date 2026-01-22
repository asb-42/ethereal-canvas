"""
Backend abstraction for Ethereal Canvas.

Provides a common interface for all generation modes
and enforces proper separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch

class GenerationBackend(ABC):
    """Abstract base class for all generation backends."""
    
    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def load(self) -> None:
        """Load the model and prepare for generation."""
        pass
    
    @abstractmethod
    def generate(self, **kwargs) -> Path:
        """
        Perform a generation step and return output path.
        
        Returns:
            Path: Path to the generated file
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources and free memory."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model and configuration information."""
        pass
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported generation modes."""
        return []
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Validate generation parameters."""
        return {"valid": True, "errors": []}


class InpaintingBackend(GenerationBackend):
    """Base class for inpainting backends."""
    
    @abstractmethod
    def inpaint(self, **kwargs) -> Path:
        """
        Perform inpainting step and return output path.
        
        Args:
            image: Base image
            mask: Inpainting mask
            prompt: Text description for masked areas
            **kwargs: Additional parameters
        
        Returns:
            Path: Path to the inpainted file
        """
        pass
    
    def get_supported_modes(self) -> List[str]:
        return ["inpaint"]