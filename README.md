# Ethereal Canvas

A modular, Unix-style image generation and image editing framework
using Gradio as UI and Qwen-Image as the initial backend.

## Features (Stage 1)
- Text to Image generation
- Image to Image editing
- Deterministic generation via seed control
- Metadata embedding
- Markdown-based logging
- Local Git-based audit trail
- Network-accessible Gradio UI
- Pinokio one-click installer

## Architecture
The application is composed of small, isolated modules with
well-defined responsibilities.

See `docs/architecture.md` for details.

## License
The Ethereal Canvas code in this repository is released under the **GNU AGPL v3**
(or later) - see `LICENSE`. That copyleft covers **this application only**.

**Model weights are not part of this release and are not redistributed by it.**
Every checkpoint is fetched separately by the person running the application,
from its own upstream, under that upstream's own terms - for example the
Qwen-Image 2.x releases under the Qwen Research License Agreement. Accepting
the AGPL for this code does not license the weights, and nothing here grants
you those weights.

Before any download the app requires a recorded, per-user acceptance of the
relevant upstream terms; see `scripts/accept_model_license.py`. Because this app
is AGPL and a Gradio server, note AGPL section 13: if you modify it and let
others use it over a network, you owe them the corresponding source.
