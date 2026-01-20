"""
Prompt management and normalization.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import random
import time


@dataclass
class PromptObject:
    prompt_text: str
    seed: int
    params: Dict
    timestamp: float


def generate_seed(base: Optional[int] = None) -> int:
    return base if base is not None else random.randint(0, 2**32 - 1)


def normalize_prompt(raw_prompt: str, seed: Optional[int] = None, params: Optional[Dict] = None) -> PromptObject:
    return PromptObject(
        prompt_text=raw_prompt.strip(),
        seed=generate_seed(seed),
        params=params or {},
        timestamp=time.time()
    )


def validate_prompt(prompt_obj: PromptObject) -> bool:
    return len(prompt_obj.prompt_text) > 0