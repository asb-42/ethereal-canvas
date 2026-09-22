# Ethereal Canvas

A modular, Unix-style image generation and editing framework:
Gradio UI, one shared pipeline, Qwen-Image 2.1 as the default backend.

## What it does

- **Text-to-image** with size presets (512²–2048², 16:9/9:16), seeded
  reproducibility, and live step progress in the Status Log.
- **Image editing** (single to multi-reference), same pipeline, no second
  model load.
- **Text-encoder switcher**: stock, or the Heretic abliteration for prompts
  the stock encoder refuses. Swaps in place, either direction.
- **Prompt-expansion toggle** (PE-T2I companion, opt-in): upstream's
  documented best-quality path, `Auto` size lets it recommend dimensions.
- Deterministic seeds, metadata sidecars, markdown run log.

## Requirements

- NVIDIA GPU around ~120 GiB (GB10 class): full-pipeline inference peaks
  near 95–119 GiB depending on allocator config; see `docs/user_guide.md`.
- The pinned `.venv-qi21` environment: `./scripts/setup_qi21_env.sh`.
- Upstream weight licences accepted per machine:
  `scripts/accept_model_license.py --model Qwen/Qwen-Image-2.1 --yes`
  (plus Heretic / PE-T2I if you use them). Weights are fetched separately
  and never redistributed here.

## Run

```bash
EC_PYTHON=$PWD/.venv-qi21/bin/python ./scripts/run.sh
```

UI at `http://<host>:7860`. First use loads the pipeline (~4 min).

## Layout

```
config/    model + server configuration (2.1 is the default)
docs/      user_guide.md, architecture.md, developer_guide.md + archive/
modules/   backends, job_runner, memory, prompt_engine, runtime, ui_gradio, ...
scripts/   run, setup_qi21_env, accept_model_license, download_models,
           validate_qi21(.sh) + archive/
tests/     backend suite: python tests/test_qi21_backend.py
```

See `docs/architecture.md` for the module map, `docs/user_guide.md` for
operation, measured costs, and troubleshooting.

## License

This application's code is **GNU AGPL v3** (or later) - see `LICENSE`.
That covers **this code only**. Model weights are fetched separately by
the operator under their own upstream terms (e.g. the Qwen Research
License Agreement); nothing here grants them. Per-user acceptance is
enforced before any download. AGPL section 13 applies over the network:
serve a modified copy and you owe its users the source.
