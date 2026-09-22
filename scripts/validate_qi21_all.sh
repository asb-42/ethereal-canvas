#!/usr/bin/env bash
# Section 7 matrix on the GB10.
# TORCH_DISABLE_NATIVE_JIT=1 is required on this box: torch's native Triton
# override of aten::bmm needs Python.h, which is absent (no python3.12-dev).
# Without it the pipeline dies at the first denoise step.
set -u
cd /srv/coding/ethereal-canvas
export EC_ALLOW_UNVALIDATED_QI21=1
export EC_MODEL_LICENSE_ACCEPTED=1
export EC_MODEL_CONFIG=config/model_config.qi21.yaml
export TORCH_DISABLE_NATIVE_JIT=1
exec .venv-qi21/bin/python scripts/validate_qi21.py "$@"
