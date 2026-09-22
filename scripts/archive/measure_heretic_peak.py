#!/usr/bin/env python3
"""Measure peak device memory across load, Heretic swap, and inference.

Run A (this script, plain env): stock load -> Heretic swap -> 1024x1024 T2I.
Run B (same script, EC_QI21_SEQUENTIAL_CPU_OFFLOAD=1): identical, offloaded.
Peaks come from the torch-side sampler (see scripts/validate_qi21.py): report
peak, not final. Writes runtime/logs/heretic-peak.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.validate_qi21 import PeakSampler, gib

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "runtime" / "logs" / "heretic-peak.json"


def main() -> int:
    import torch
    from modules.backends.image_21 import QwenImage21Backend
    from modules.runtime.paths import load_model_config

    offload = __import__("os").environ.get(
        "EC_QI21_SEQUENTIAL_CPU_OFFLOAD", "0")
    cfg = load_model_config()
    backend = QwenImage21Backend(cfg.get("generate_model"))
    rows = []

    s = PeakSampler()
    s.start()
    t0 = time.time()
    backend.load()
    s.stop()
    rows.append({"phase": "load_stock", "wall_s": round(time.time() - t0, 1),
                 "peak_GiB": round(gib(s.peak_bytes), 3)})
    print("load done", rows[-1], flush=True)

    s = PeakSampler()
    s.start()
    t0 = time.time()
    backend._ensure_text_encoder("heretic")
    s.stop()
    rows.append({"phase": "swap_heretic", "wall_s": round(time.time() - t0, 1),
                 "peak_GiB": round(gib(s.peak_bytes), 3)})
    print("swap done", rows[-1], flush=True)

    s = PeakSampler()
    s.start()
    t0 = time.time()
    imgs = backend.generate("a red cube on a white table",
                            width=1024, height=1024)
    s.stop()
    out = Path(imgs)  # the generate() shim persists and returns the path
    rows.append({"phase": "infer_heretic_1024", "wall_s": round(time.time() - t0, 1),
                 "peak_GiB": round(gib(s.peak_bytes), 3),
                 "output": str(out)})
    print("infer done", rows[-1], flush=True)

    payload = {
        "offload": offload,
        "gpu": torch.cuda.get_device_name(0),
        "phases": rows,
    }
    EVIDENCE.write_text(json.dumps(payload, indent=2))
    print(f"evidence: {EVIDENCE}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
