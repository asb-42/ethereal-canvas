"""
Backend for the unified Qwen-Image 2.1 checkpoint.

One checkpoint serves text-to-image, image editing, multi-reference composition
and RGBA output, which is why this class deliberately spans the roles the
2512/2511 pair splits across two backends.

It needs the pinned development build of diffusers held in .venv-qi21 (see
requirements_qi21.txt): the 0.36 release has no QwenImage21Pipeline at all.
Every diffusers import is therefore deferred to load() and this module stays
importable under the older interpreter, so a checkout running the legacy stack
can still list the tab and fail with an actionable message instead of failing
to import.
"""
import os
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

import logging

from modules.runtime import model_access


logger = logging.getLogger(__name__)

PIPELINE_CLASS = "QwenImage21Pipeline"

#: The card's own recommended working resolution for edits. The library default
#: is 1024, well below what the checkpoint is tuned for.
RECOMMENDED_EDIT_RESOLUTION = 2048
LIBRARY_DEFAULT_RESOLUTION = 1024

#: Drop-in text encoder variants for the 2.1 pipeline. "stock" is the encoder
#: shipped inside Qwen/Qwen-Image-2.1; "heretic" is the community abliteration
#: (refusal removed, same shapes/dtype), fetched separately and swapped into
#: the live pipeline per generation.
HERETIC_MODEL_ID = "pottokao/Qwen-Image-2.1-Text-Encoder-Heretic"
TEXT_ENCODER_VARIANTS = ("stock", "heretic")

#: The card fixes both of these; the library defaults already agree with them,
#: but they are stated so a future library default change cannot silently
#: undo the tuning.
NUM_INFERENCE_STEPS = 40
TRUE_CFG_SCALE = 1.0


def _env_flag(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).strip().lower() in (
        "1", "true", "yes", "on")


class QwenImage21Backend:
    """Unified generation and editing against Qwen-Image 2.1."""

    #: Naming variants the adapter checks, for consistency with its siblings.
    SUPPORTS_NEGATIVE_PROMPT = True
    supports_negative_prompt = True

    def __init__(self, model_name: str = "Qwen/Qwen-Image-2.1",
                 device: str = "cuda", dtype: str = "bfloat16"):
        self.model_name = model_name
        self.pipeline = None
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(torch, dtype, torch.bfloat16)
        self.is_loaded = False
        # Which text encoder the live pipeline currently carries. Swapped per
        # generation by _ensure_text_encoder; "stock" is what from_pretrained
        # puts there, "heretic" the abliterated drop-in.
        self.text_encoder_variant = "stock"
        # The companion prompt-expansion model, loaded on first use only when the
        # feature is switched on. Kept here rather than module-global so two
        # backends do not silently share a GPU resident 9B.
        self._prompt_rewriter = None

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """
        Fetch and prepare the checkpoint.

        Refuses before touching the network unless the upstream terms have been
        accepted on this machine; see scripts/accept_model_license.py.
        """
        model_access.require(self.model_name)

        import diffusers
        pipeline_class = getattr(diffusers, PIPELINE_CLASS, None)
        if pipeline_class is None:
            raise RuntimeError(
                f"this checkout cannot serve {self.model_name}: the installed "
                f"diffusers ({diffusers.__version__}) has no {PIPELINE_CLASS}. "
                f"Build the pinned environment first: ./scripts/setup_qi21_env.sh "
                f"and launch with EC_PYTHON=.venv-qi21 ./scripts/run.sh")

        logger.info(f"Loading {PIPELINE_CLASS} {self.model_name}")
        # cache_dir is what makes a pre-fetched checkpoint count. Every sibling
        # backend passes it; without it from_pretrained resolves the repo id
        # against the ambient hub cache, so a weights tree placed under
        # models/Qwen-Image-2.1 is invisible and the load silently re-fetches
        # ~31 GiB over the network instead of using what is already on disk.
        from modules.runtime.paths import model_cache_dir_for

        cache_dir = model_cache_dir_for(self.model_name)
        if cache_dir.is_dir():
            logger.info(f"using local weights at {cache_dir}")
        self.pipeline = pipeline_class.from_pretrained(
            self.model_name, cache_dir=str(cache_dir), torch_dtype=self.dtype)
        self.pipeline = self.pipeline.to(self.device)

        # The DiT attention path is the sharpest memory spike on a single GPU.
        self.pipeline.enable_attention_slicing()

        # Free hand over VRAM: stage components on the CPU in dependency order.
        # This snapshot's enable_model_cpu_offload takes no tuning kwargs
        # (no weights_on_gpu / memory-reserve / order arguments exist here),
        # so call it bare; extra kwargs die with TypeError at load.
        #
        # Measured verdict on the GB10 (unified 128 GB package, 1024 squared,
        # Heretic encoder): offload changes the peak from 118.3 to 119.0 GiB
        # (noise) while slowing inference ~6x (50s -> 315s). CPU and GPU share
        # the same memory here, so relocating weights cannot move the
        # device-side counter. Leave this OFF on unified-memory boxes; it
        # exists for discrete GPUs with small VRAM and large system RAM.
        if _env_flag("EC_QI21_SEQUENTIAL_CPU_OFFLOAD"):
            self.pipeline.enable_model_cpu_offload()
            logger.info("Sequential CPU offload enabled")

        self.is_loaded = True
        logger.info(f"{self.model_name} loaded on {self.device}")

        # Honour a non-stock encoder requested for the whole launch, so a box
        # that only ever serves Heretic pays the swap once, not per call.
        want = os.environ.get("EC_QI21_TEXT_ENCODER", "stock").strip().lower()
        if want and want != "stock":
            self._ensure_text_encoder(want)
        return True

    # ------------------------------------------------------- text encoder swap
    def _ensure_text_encoder(self, variant: str) -> None:
        """Carry ``variant`` ("stock" or "heretic") on the live pipeline.

        Stock needs no work: it is what from_pretrained installed. Heretic is
        loaded from the local snapshot and swapped in place, so switching costs
        one 17.5 GiB encoder load, not a second full pipeline. Anything
        unrecognised, unlicensed, or absent from disk raises with the command
        that fixes it rather than silently rendering with the wrong encoder.
        """
        want = (variant or "stock").strip().lower()
        if want not in TEXT_ENCODER_VARIANTS:
            raise ValueError(
                f"unknown text encoder {variant!r}; "
                f"choose one of {list(TEXT_ENCODER_VARIANTS)}")
        if want == self.text_encoder_variant:
            return
        if self.pipeline is None and not self.load():
            raise RuntimeError("pipeline unavailable after load")

        if want == "stock":
            raise RuntimeError(
                "switching back to the stock encoder needs a fresh load: "
                "restart the UI (or release() this backend) rather than "
                "re-fetching 17.5 GiB of stock weights that were overwritten "
                "in memory")
        model_access.require(HERETIC_MODEL_ID)

        from modules.backends.prompt_rewriter import _local_snapshot
        snap = _local_snapshot(HERETIC_MODEL_ID, require_weights=True)
        if snap is None:
            raise RuntimeError(
                f"Heretic encoder is not on disk; fetch it with: "
                f"python scripts/download_models.py --model {HERETIC_MODEL_ID} "
                f"(after accepting its terms: "
                f"python scripts/accept_model_license.py "
                f"--model {HERETIC_MODEL_ID} --yes)")

        from transformers import Qwen3VLForConditionalGeneration
        # Free the stock encoder BEFORE loading Heretic: the old order held
        # both (~34 GiB) plus the DiT at once, which is the highest transient
        # peak in the app on a 121.6 GiB device. Dropping the reference and
        # emptying the cache first keeps the swap under the inference peak.
        import gc
        old = self.pipeline.text_encoder
        self.pipeline.text_encoder = None
        del old
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()
        logger.info(f"Loading Heretic text encoder from {snap}")
        enc = Qwen3VLForConditionalGeneration.from_pretrained(
            str(snap), dtype=self.dtype,
            device_map="auto", local_files_only=True).eval()

        self.pipeline.text_encoder = enc.to(self.device) \
            if self.device == "cpu" else enc
        del enc
        if _env_flag("EC_QI21_SEQUENTIAL_CPU_OFFLOAD"):
            self.pipeline.enable_model_cpu_offload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.text_encoder_variant = "heretic"
        logger.info("Heretic text encoder now serving")

    # ------------------------------------------------------------- generation
    # --------------------------------------------------------------------------
    # Prompt rewriting
    # --------------------------------------------------------------------------
    def _rewriter(self):
        """The shared rewriter for this backend, or None when it is switched off.

        The instance is held on the backend so a second generation does not pay
        for a second 9B load, and so memory accounting can see it.
        """
        from modules.backends.prompt_rewriter import PromptRewriter

        if self._prompt_rewriter is None:
            self._prompt_rewriter = PromptRewriter()
        return self._prompt_rewriter

    def _maybe_rewrite(self, prompt: str,
                       width: Optional[int], height: Optional[int],
                       force: Optional[bool] = None):
        """Expand ``prompt`` through the companion rewriter when it is enabled.

        Upstream ships the 2.1 release with a separate prompt-expansion model and
        documents it as the way to get the release's best output; the image
        pipeline itself has no hook for it, so consulting it is this backend's
        job. It is opt-in because it is a second model resident on the GPU.

        ``force`` is the per-generation UI toggle (True/False); when None the
        launch-wide EC_QI21_PROMPT_REWRITER / config switch decides.

        An explicit ``width``/``height`` from the caller always wins over the
        rewriter's recommendation: asking for a size is a decision the rewriter
        should not undo.

        Any failure returns the original prompt. A text model having a bad day
        must not cost the user an image.

        The rewriter is released after every use rather than kept resident:
        18 GiB held through a ~118 GiB inference peak does not fit a
        121.6 GiB device, so each rewrite pays a reload instead of risking OOM.
        """
        from modules.backends import prompt_rewriter as pr

        active = force if force is not None else pr.rewriter_enabled()
        if not active:
            return prompt, width, height

        rw = self._rewriter()
        if not rw.available:
            logger.warning(
                f"prompt rewriter enabled but {rw.model_id} is not on disk; "
                f"generating from the prompt as written")
            return prompt, width, height

        result = rw.rewrite(prompt)
        release = getattr(rw, "release", None)
        if callable(release):
            try:
                release()
            except Exception:
                pass
            if self._prompt_rewriter is rw:
                self._prompt_rewriter = None
        if result is None:
            return prompt, width, height

        logger.info(
            f"prompt rewritten ({len(result.prompt)} chars, "
            f"ratio={result.wh_ratio or 'none'})")
        if width is None and height is None and result.width:
            return result.prompt, result.width, result.height
        return result.prompt, width, height

    def generate_image(self, prompt: str,
                       negative_prompt: Optional[str] = None,
                       width: Optional[int] = None,
                       height: Optional[int] = None,
                       num_steps: int = NUM_INFERENCE_STEPS,
                       seed: Optional[int] = None,
                       num_images_per_prompt: int = 1,
                       output_resolution: Optional[int] = None,
                       output_type: str = "pil",
                       text_encoder: Optional[str] = None,
                       prompt_rewrite: Optional[str] = None,
                       progress_cb: Any = None) -> List[Any]:
        """Synthesise images from text alone."""
        if isinstance(prompt, (list, tuple)) and len(prompt) > 1:
            raise ValueError(
                "Qwen-Image 2.1 applies every image in `image` to every prompt, "
                "so a multi-prompt batch has no defined pairing. Generate one "
                "prompt at a time.")

        force = None
        if prompt_rewrite is not None:
            force = str(prompt_rewrite).strip().lower() in (
                "1", "true", "yes", "on")
        prompt, width, height = self._maybe_rewrite(
            prompt, width, height, force=force)

        return self._call(
            prompt=prompt, image=None, negative_prompt=negative_prompt,
            num_steps=num_steps, seed=seed, width=width, height=height,
            num_images_per_prompt=num_images_per_prompt,
            output_resolution=output_resolution, output_type=output_type,
            text_encoder=text_encoder, progress_cb=progress_cb)

    # ---------------------------------------------------------------- editing
    def edit_images(self, image_sources: List[Image.Image], prompt: str,
                    strength: float = None,
                    negative_prompt: Optional[str] = None,
                    num_steps: int = NUM_INFERENCE_STEPS,
                    seed: Optional[int] = None,
                    reference_images: Optional[List[Image.Image]] = None,
                    output_resolution: Optional[int] = None,
                    output_type: str = "pil",
                    text_encoder: Optional[str] = None,
                    progress_cb: Any = None) -> List[Any]:
        """
        Edit, or compose from, one to ten reference images.

        `strength` exists only for signature compatibility with the 2511 edit
        backend: this is a flow-matching model with a fixed schedule and no
        denoising-strength knob, so a value here is reported and ignored.
        """
        images: List[Image.Image] = [
            img for img in (image_sources or []) if img is not None]
        if reference_images:
            images += [img for img in reference_images if img is not None]
        if not images:
            raise ValueError(
                "no source image supplied; Qwen-Image 2.1 editing needs at "
                "least one reference image")
        if len(images) > 10:
            raise ValueError(
                f"{len(images)} reference images given, the checkpoint takes at "
                f"most 10")
        if strength is not None:
            logger.info(
                f"strength={strength} ignored: Qwen-Image 2.1 is a "
                f"flow-matching model with a fixed schedule")

        return self._call(
            prompt=prompt, image=images if len(images) > 1 else images[0],
            negative_prompt=negative_prompt, num_steps=num_steps, seed=seed,
            output_resolution=output_resolution or RECOMMENDED_EDIT_RESOLUTION,
            output_type=output_type, text_encoder=text_encoder,
            progress_cb=progress_cb)

    # ------------------------------------------------------------- shared call
    def _call(self, **kwargs: Any) -> List[Any]:
        """Run the pipeline, passing only the keyword arguments it accepts."""
        variant = kwargs.pop("text_encoder", None)
        progress_cb = kwargs.pop("progress_cb", None)
        if self.pipeline is None and not self.load():
            raise RuntimeError("pipeline unavailable after load")
        if variant:
            # Per-generation encoder choice from the UI switcher. A miss here
            # must be loud: silently rendering Heretic-labelled output with
            # the stock encoder (or vice versa) is the one failure mode that
            # matters for this feature.
            self._ensure_text_encoder(variant)

        negative_prompt = kwargs.pop("negative_prompt", None)
        num_steps = kwargs.pop("num_steps", NUM_INFERENCE_STEPS)
        seed = kwargs.pop("seed", None)

        call_kwargs: Dict[str, Any] = {
            "num_inference_steps": int(num_steps),
            # Flow matching: values above 1.0 degrade output. The former edit
            # backend's guidance defaults must not leak in here.
            "true_cfg_scale": TRUE_CFG_SCALE,
            **kwargs,
        }
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        # An explicit None must not clobber a pipeline default. The UI path
        # sends output_resolution=None (no size control in the form yet), and
        # the pipeline resolves sizes as `width or output_resolution` with a
        # default of 1024. Passing None through turns both into None and dies
        # in check_inputs with `None % int`.
        for key in ("output_resolution", "width", "height"):
            if call_kwargs.get(key) is None:
                call_kwargs.pop(key, None)

        if seed is not None:
            # torch.manual_seed seeds globally and takes no generator; the
            # per-call path is a Generator seeded itself, which is also what
            # makes the result reproducible.
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            call_kwargs["generator"] = generator
            # The card ties bit-for-bit reproducibility to disabling the KV
            # cache, so pin it off whenever a seed is in play.
            call_kwargs["use_kv_cache"] = _env_flag("EC_QI21_KV_CACHE", False)

        if progress_cb is not None:
            # Real step progress for the UI: the pipeline calls this after
            # every denoising step. It must return the kwargs dict untouched
            # (the loop pops latents/prompt_embeds out of it), and it must
            # never raise - a progress reporter must not kill a render.
            # Deliberately step counts only, no latent previews: a VAE decode
            # per step costs seconds and gigabytes on this device.
            total_steps = int(num_steps)

            def _step_cb(pipe, i, t, cb_kwargs):
                try:
                    progress_cb(int(i) + 1, total_steps)
                except Exception:
                    pass
                return cb_kwargs

            call_kwargs["callback_on_step_end"] = _step_cb

        # Iterating Signature.parameters yields names, not Parameter objects.
        import inspect as _inspect
        valid = set(self.pipeline.__call__.__annotations__) | set(
            _inspect.signature(self.pipeline.__call__).parameters)
        dropped = set(call_kwargs) - valid
        if dropped:
            logger.warning(f"dropping unsupported arguments: {sorted(dropped)}")
            call_kwargs = {k: v for k, v in call_kwargs.items()
                           if k in valid}

        try:
            result = self.pipeline(**call_kwargs)
        except ValueError as exc:
            # Precomputed prompt embeddings cannot be combined with the image
            # padding mask path; surface that rather than masking it.
            raise RuntimeError(f"Qwen-Image 2.1 generation failed: {exc}") from exc

        images = getattr(result, "images", None)
        if images is None:
            images = getattr(result, "images_list", None) or []
        return list(images)

    # ------------------------------------------------------- adapter contract
    # The adapter (and through it the UI) speaks the legacy pair's language:
    # generate(prompt) / edit(prompt, input_path) returning a file path.
    # This backend natively speaks generate_image / edit_images returning PIL
    # images, so these shims translate. Without them the 2.1 route dies with
    # AttributeError the moment the UI calls it.
    def _persist(self, image: Image.Image, prefix: str) -> str:
        from modules.runtime.paths import OUTPUTS_DIR, timestamp

        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / f"{prefix}_{timestamp()}.png"
        image.save(path)
        return str(path)

    def generate(self, prompt: str, seed: Optional[int] = None,
                 progress_cb: Any = None, **kwargs: Any) -> str:
        """Legacy-contract wrapper around generate_image."""
        images = self.generate_image(
            prompt=prompt, seed=seed, progress_cb=progress_cb, **kwargs)
        if not images:
            raise RuntimeError("Qwen-Image 2.1 returned no images")
        return self._persist(images[0], "qi21_t2i")

    def edit(self, prompt: str, input_path: Union[str, Any],
             seed: Optional[int] = None, progress_cb: Any = None,
             **kwargs: Any) -> str:
        """Legacy-contract wrapper around edit_images."""
        path = (input_path.name if hasattr(input_path, "name")
                else str(input_path))
        with Image.open(path) as img:
            ref = img.convert("RGB")
            ref.load()
        images = self.edit_images([ref], prompt, seed=seed,
                                    progress_cb=progress_cb, **kwargs)
        if not images:
            raise RuntimeError("Qwen-Image 2.1 returned no images")
        return self._persist(images[0], "qi21_edit")

    # ---------------------------------------------------------------- release
    def release(self) -> None:
        """Free the checkpoint, including the components the offload keeps pinned."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "synchronize"):
                torch.cuda.synchronize()
