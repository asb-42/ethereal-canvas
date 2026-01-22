import pytest
from modules.job_runner.runner import execute_task
from modules.prompt_engine.engine import normalize_prompt

def test_empty_prompt():
    with pytest.raises(ValueError):
        execute_task("generate", "", seed=42)

def test_invalid_image():
    with pytest.raises(FileNotFoundError):
        execute_task("edit", "A red bicycle", seed=42, input_path="nonexistent.jpg")

def test_unavailable_gpu():
    """Test behavior when GPU is not available."""
    import sys
    from unittest.mock import patch
    
    # Mock torch.cuda to simulate no GPU available
    with patch.dict('sys.modules', {'torch.cuda': None}):
        with patch('torch.cuda.is_available', return_value=False):
            # Test that backend falls back to CPU
            from modules.backends.text_to_image import TextToImageBackend
            backend = TextToImageBackend()
            
            # Should default to CPU when CUDA is not available
            assert backend.device == "cpu"
            
            # Should still be able to load (with stub implementation)
            backend.load()
            assert backend.loaded is True

def test_invalid_task_type():
    with pytest.raises(ValueError):
        execute_task("invalid", "test", seed=42)