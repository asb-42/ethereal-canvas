"""
GPU memory guardrails.
"""

import torch

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()