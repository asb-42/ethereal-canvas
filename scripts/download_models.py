#!/usr/bin/env python3
"""
Model downloader utility for Ethereal Canvas.
Downloads and verifies Qwen models from HuggingFace.

Weights belong to the upstream and are licensed by them, so nothing here
reaches the network until the operator has recorded an acceptance for that
specific model; see scripts/accept_model_license.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import hashlib
from datetime import datetime

#: This script is run from the repository root, which is not on sys.path when it
#: is executed as a file, so the package imports below would not resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.runtime import model_access
from modules.runtime import paths


def log_message(message: str):
    """Simple logging function."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


#: What a complete checkpoint must contain, per release.
#:
#: The 2.1 release is a diffusers component tree: model_index.json at the top
#: and one sub-directory per component holding its own config.json plus sharded
#: safetensors. It has no top-level config.json, tokenizer.json or
#: tokenizer_config.json at all -- those live under processor/, vae/ and
#: text_encoder/ -- so the single legacy checklist rejected a perfectly good
#: 2.1 download every time. The legacy pair keeps the older file list.
REQUIRED_FILES = {
    "legacy": ["config.json", "tokenizer.json", "tokenizer_config.json"],
    # Either marker is enough, and both are accepted. The image checkpoints are
    # diffusers and carry model_index.json at the root with their config.json
    # down inside transformer/; the prompt-rewriter checkpoints are plain
    # transformers and carry a root config.json with no index at all. Requiring
    # the index alone would reject a fully downloaded rewriter.
    "qi21": ["model_index.json", "config.json"],
}

#: Weight file extensions a checkpoint must carry at least one of.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".ckpt", ".pth")


def _is_qi21(model_name: str) -> bool:
    return paths.is_qwen_image_21(model_name)


def verify_model_directory(model_dir: Path, model_name: str = "") -> bool:
    """Verify that a model directory holds a complete, usable checkpoint."""
    if not model_dir.exists():
        return False

    required = REQUIRED_FILES["qi21" if _is_qi21(model_name) else "legacy"]

    if _is_qi21(model_name):
        # The two 2.1 layouts are alternatives, not a conjunction: see the note on
        # REQUIRED_FILES. Requiring both files would reject every 2.1 checkpoint.
        if not any((model_dir / name).exists() for name in required):
            log_message(
                "Missing checkpoint marker: need "
                + " or ".join(required) + f" in {model_dir}")
            return False
    else:
        # Check for essential files
        for file_name in required:
            if not (model_dir / file_name).exists():
                log_message(f"Missing essential file: {file_name}")
                return False

    # A checkpoint is only complete once its weights are present. Checking for
    # the config files alone would call a half-fetched tree "verified".
    weights = [p for p in model_dir.rglob("*")
               if p.is_file() and p.name.endswith(WEIGHT_SUFFIXES)]
    if not weights:
        log_message("No weight files (.safetensors/.bin/.ckpt/.pth) present")
        return False

    # Check for incomplete files
    incomplete_files = list(model_dir.rglob("*.incomplete"))
    if incomplete_files:
        log_message(f"Found {len(incomplete_files)} incomplete files")
        return False

    total = sum(p.stat().st_size for p in weights)
    log_message(f"✅ Model directory verified: {model_dir} "
                f"({len(weights)} weight files, {total / 2**30:.2f} GiB)")
    return True


def download_qwen_model(model_name: str, cache_dir: Path) -> bool:
    """Download a Qwen model using huggingface_hub."""
    # Weights belong to the upstream and are licensed by them: refuse before any fetch.
    model_access.require(model_name)
    try:
        from huggingface_hub import snapshot_download

        log_message(f"Starting download of {model_name}...")
        log_message(f"Cache directory: {cache_dir}")

        # Create cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Download model with progress
        start_time = time.time()

        downloaded_path = snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )

        download_time = time.time() - start_time
        log_message(f"Download completed in {download_time:.2f} seconds")

        # Verify download
        if verify_model_directory(Path(downloaded_path), model_name):
            log_message(f"✅ Successfully downloaded and verified {model_name}")
            return True
        else:
            log_message(f"❌ Verification failed for {model_name}")
            return False

    except ImportError:
        log_message("❌ huggingface_hub or transformers not installed")
        log_message("Please install: pip install huggingface_hub transformers")
        return False
    except Exception as e:
        log_message(f"❌ Failed to download {model_name}: {e}")
        return False


#: Every model this application knows how to fetch, in the order the original
#: all-models sweep used.
KNOWN_MODELS = [
    "Qwen/Qwen-Image-2512",        # Text-to-image model
    "Qwen/Qwen-Image-Edit-2511",    # Image editing model
    "Qwen/Qwen-Image-2.1",          # Unified 2.1 release (needs .venv-qi21)
    # The 2.1 companions. They are separate checkpoints rather than parts of the
    # image model: the pipeline exposes no hook for them, so they are loaded
    # alongside it by modules/backends/prompt_rewriter.py. They are listed here so
    # the sanctioned downloader and its licence gate cover them; they are kept out
    # of the default all-models sweep below, because pulling 35 GiB of optional
    # text models for someone who never enabled the feature is not a courtesy.
    "Qwen/Qwen-Image-2.1-PE-T2I",   # Prompt rewriter, text to image
    "Qwen/Qwen-Image-2.1-PE-I2I",   # Prompt rewriter, image editing
]

#: What an unqualified run fetches. Optional companions stay opt-in.
DEFAULT_MODELS = [m for m in KNOWN_MODELS if "-PE-" not in m]


def main(argv=None) -> int:
    """Main function to download required models.

    ``--model`` selects one release. It used to be accepted and then ignored:
    main() took no arguments and read no argv, so asking for the 2.1 release
    still swept all three checkpoints, each behind its own licence acceptance.
    Selecting one model now fetches exactly that one.
    """
    parser = argparse.ArgumentParser(
        description="Download Qwen model weights for Ethereal Canvas.")
    parser.add_argument("--model",
                        help="single model to fetch, e.g. Qwen-Image-2.1")
    parser.add_argument("--list", action="store_true",
                        help="list the fetchable models and exit")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the cache already verifies")
    args = parser.parse_args(argv)

    if args.list:
        for model_name in KNOWN_MODELS:
            cache = paths.model_cache_dir_for(model_name)
            state = "present" if verify_model_directory(cache, model_name) else "absent"
            print(f"{model_name:32s} {state:8s} {cache}")
        return 0

    if args.model:
        from modules.runtime.model_access import normalize_model_id
        key = normalize_model_id(args.model)
        if key not in KNOWN_MODELS:
            log_message(f"❌ Unknown model {args.model!r}. Known models:")
            for model_name in KNOWN_MODELS:
                log_message(f"    {model_name}")
            return 2
        wanted = [key]
    else:
        log_message("🚀 Starting model download process (all known models)...")
        wanted = list(DEFAULT_MODELS)

    success_count = 0

    for model_name in wanted:
        model_cache_dir = paths.model_cache_dir_for(model_name)

        # Check if model already exists and is verified
        if not args.force and verify_model_directory(model_cache_dir, model_name):
            log_message(f"✅ {model_name} already exists and verified")
            success_count += 1
            continue

        # Download the model
        if download_qwen_model(model_name, model_cache_dir):
            success_count += 1

        log_message("-" * 50)

    # Summary
    log_message(f"📊 Download complete: {success_count}/{len(wanted)} models ready")

    if success_count == len(wanted):
        log_message("🎉 All models are ready for use!")
        return 0
    log_message("⚠️  Some models failed to download. Check the logs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
