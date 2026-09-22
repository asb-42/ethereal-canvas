# User Guide

## What it runs

Ethereal Canvas serves the unified **Qwen-Image 2.1** checkpoint for both
tabs: text-to-image generation and image editing. One 31 GiB download, one
pipeline, shared across tabs after the first load.

Requirements: NVIDIA GPU with ~120 GiB (GB10 class), the `.venv-qi21`
environment (`scripts/setup_qi21_env.sh`), and an accepted upstream licence
(`scripts/accept_model_license.py --model Qwen/Qwen-Image-2.1 --yes`).

## Launch

```bash
EC_PYTHON=$PWD/.venv-qi21/bin/python ./scripts/run.sh
```

UI at `http://<host>:7860`. First use of either tab loads the pipeline
(~4 min); it stays resident afterwards.

## Generate tab

- **Prompt** + optional **seed** (same prompt + seed = identical pixels).
- **Text encoder**: Stock, or Heretic (abliterated community encoder for
  prompts the stock encoder refuses; needs its own 17.5 GiB download).
  Switching back to stock needs a UI restart.
- **Image size**: presets with measured wall times at 40 steps
  (512² ~13s … 1024² ~1 min … 2048² ~4 min). `Auto` lets the prompt
  rewriter recommend a size, else 1024².
- **Expand prompt (PE-T2I rewriter)**: 18 GiB companion model, reloaded per
  run to protect VRAM. Best quality per upstream; costs ~1–2 min extra.
- Progress streams in the Status Log (`Denoising step i/40`).

## Edit tab

Upload an image, describe the change. Same encoder switcher. Output
resolution 1024 (faster) or 2048 (card recommended, ~7 min measured).
Strength is accepted and ignored — 2.1 is flow-matching with a fixed
schedule.

## Memory reality (GB10, 121.6 GiB device)

Full pipeline inference peaks ~118–119 GiB. Sequential CPU offload does
not help on unified-memory boxes (measured: same peak, 6x slower) and
defaults off. If a run dies, the Status Log names the cause; the most
common transient is holding two encoders at once, which the app avoids by
freeing before swapping.

## Troubleshooting

- **Port in use**: `PORT=7861 ... ./scripts/run.sh`.
- **Licence refusal**: run the accept command above; acceptances live in
  gitignored `runtime/model_license_acceptances.json`, one per machine.
- **Triton/gcc failure at step 1**: the launcher already sets
  `TORCH_DISABLE_NATIVE_JIT=1`; if you launch otherwise, set it yourself.
- Outputs land in `runtime/outputs/` (gitignored).
