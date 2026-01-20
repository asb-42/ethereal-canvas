"""
Simple latency and memory benchmark.
"""

import time
import torch

def benchmark(backend, prompt):
    start = time.time()
    backend.generate_image(prompt)
    elapsed = time.time() - start
    mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    return elapsed, mem