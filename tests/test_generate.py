from modules.job_runner.runner import execute_task

def test_basic_generate():
    out = execute_task("generate", "A red bicycle on a beach", seed=42)
    assert out.endswith(".png")