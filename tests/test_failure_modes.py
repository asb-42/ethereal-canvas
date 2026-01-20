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
    # This test would need mocking in real implementation
    pass

def test_invalid_task_type():
    with pytest.raises(ValueError):
        execute_task("invalid", "test", seed=42)