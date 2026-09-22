#!/usr/bin/env bash
# Fetch the prompt-rewriter checkpoint through the sanctioned downloader.
set -u
cd /srv/coding/ethereal-canvas
export HF_HUB_DISABLE_XET=1
export EC_MODEL_LICENSE_ACCEPTED=1
export EC_MODEL_CONFIG=config/model_config.qi21.yaml
exec .venv-qi21/bin/python scripts/download_models.py --model Qwen/Qwen-Image-2.1-PE-T2I
