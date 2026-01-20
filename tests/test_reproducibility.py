from modules.job_runner.runner import execute_task

def test_same_seed_same_output():
    out1 = execute_task("generate", "A blue cube", seed=123)
    out2 = execute_task("generate", "A blue cube", seed=123)
    assert open(out1, "rb").read() == open(out2, "rb").read()