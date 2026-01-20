# Architecture Overview

This project follows the Unix philosophy:
small programs, single responsibility, explicit interfaces.

## Core Modules
- img_read
- img_write
- prompt_engine
- model_adapter
- qwen_image_backend
- job_runner
- ui_gradio
- logging

Each module can be tested, debugged, and replaced independently.

## Model Abstraction
All image models must implement the same adapter interface.
This allows swapping the backend without changing UI or orchestration.

## Logging & Reproducibility
Every execution step:
- is logged in Markdown
- is committed to Git
- contains prompt, seed, model version, and output reference