"""
Prompt rewriter for the Qwen-Image 2.1 text-to-image path.

Qwen-Image 2.1 ships a companion checkpoint, ``Qwen/Qwen-Image-2.1-PE-T2I``, that
is not part of the image model and is not reachable through the image pipeline:
it is a fine-tuned Qwen3.5-VL 9B causal language model that takes a brief image
request in any language and returns a detailed English prompt plus a recommended
aspect ratio. Upstream documents it as the way to get the release's best output,
so a generation path that never consults it is quietly using the model's
degraded mode.

It is a separate download under its own card, so it goes through the same
licence gate as every other checkpoint.

Design notes, each one a thing this module had to establish rather than assume:

* **It is a transformers model, not a diffusers one.** The image pipeline's
  ``__call__`` exposes no rewriter hook of any kind, so this cannot be turned on
  by passing a flag; it is a genuine second model loaded alongside the first.
* **``enable_thinking`` is not in ``apply_chat_template``'s signature.** It is
  read by the shipped Jinja template through ``is defined``, so it travels in
  ``kwargs`` and does have an effect. Dropping it because a signature listing
  does not show it would silently change what the model is asked to do.
* **The answer is JSON behind a reasoning block.** It is split on the ``THINK_CLOSE`` separator
and parsed, and a parse failure is a skipped rewrite rather
  than a failed generation: a rewriter that can break image generation is worse
  than no rewriter, because the user loses an image to a text model's bad day.
* **Weights are never fetched as a side effect of generating.** If the
  checkpoint is not on disk the rewrite is skipped and the operator is told the
  command that would put it there.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Default rewriter checkpoint for the text-to-image route.
PE_T2I_MODEL_ID = "Qwen/Qwen-Image-2.1-PE-T2I"

#: Aspect ratios the rewriter is documented to recommend, mapped to the pixel
#: sizes the image model renders them at. Taken verbatim from the upstream
#: model card's diffusers integration so the two agree on what "16:9" means.
WH_RATIO_TO_SIZE = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

#: Fallback used when the rewriter returns a ratio this table has never heard
#: of. Deliberately the square 2048 pair, matching the card's own default.
DEFAULT_SIZE = (2048, 2048)

#: Sampling settings from the model card. The rewriter is a reasoning model and
#: these are what its published results were produced with.
GENERATE_KWARGS = {
    "max_new_tokens": 16256,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
}

#: Separator between the model's reasoning and its answer.
#:
#: Assembled from character codes rather than written out. This is a ChatML
#: control token, and the tooling that moves source around has been observed
#: rewriting it into a different ChatML token and dropping its delimiters,
#: leaving an unterminated string. Codes keep the value provable, not plausible.
THINK_CLOSE = "<|im_end|>"


@dataclass
class Rewrite:
    """What the rewriter decided about a request."""

    prompt: str
    wh_ratio: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    thinking: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def size(self):
        return (self.width, self.height)


def _local_snapshot(model_id: str, require_weights: bool = False) -> Optional[Path]:
    """The on-disk snapshot for ``model_id``, or None if it is not there.

    Never triggers a fetch. Callers decide what absence means.

    "Is there a checkpoint here" is layout dependent, and this repo holds both
    layouts. A diffusers checkpoint identifies itself with model_index.json at the
    snapshot root and keeps its config.json down inside transformer/ and vae/; a
    transformers checkpoint identifies itself with a root config.json and has no
    index at all. Asking for only one of the two therefore reports the other
    layout as missing, which is how the image model first read as incomplete here.

    ``require_weights`` exists because a snapshot can be well-formed and still be
    useless: fetching only the small files leaves a directory holding configs, the
    tokenizer and system_prompt.txt with no shards at all. A check stopping at the
    config calls that ready and then fails inside from_pretrained.
    """
    from modules.runtime import paths

    root = paths.model_cache_dir_for(model_id)
    if not root.is_dir():
        return None
    snaps = sorted(root.rglob("snapshots/*"))
    for snap in snaps:
        if not _is_checkpoint_root(snap):
            continue
        if require_weights and not _has_weights(snap):
            continue
        return snap
    return None


def _is_checkpoint_root(snap: Path) -> bool:
    """Whether ``snap`` is the root of a checkpoint in either supported layout."""
    return ((snap / "model_index.json").is_file()          # diffusers
            or (snap / "config.json").is_file())           # transformers


def _has_weights(snap: Path) -> bool:
    """Whether ``snap`` holds real weight shards, not merely a config tree.

    Size matters as well as presence: a transfer that died partway can leave a
    shard file that is a few kilobytes of stub. The floor sits far below the real
    17.55 GiB checkpoint and far above anything a partial fetch leaves behind.
    """
    floor = 1 * 1024 ** 3            # 1 GiB
    for name in ("*.safetensors", "*.bin", "*.ckpt", "*/*.safetensors"):
        for candidate in snap.rglob(name):
            try:
                if candidate.stat().st_size >= floor:
                    return True
            except OSError:
                continue
    return False


def parse_rewriter_output(text: str) -> Optional[dict]:
    """Pull the JSON answer out of a reasoning block.

    The card's own example splits on ``
</think>`` and parses what follows. That
    is the normal case. The fallbacks below exist because a sampling model
    sometimes wraps its answer in a code fence or emits prose before it, and a
    rewrite that is merely unparsable should cost the user an enhancement, not
    an image.
    """
    if not text:
        return None

    _, _, answer = text.partition(THINK_CLOSE)
    candidate = answer.strip()

    # A fence is cosmetic; strip it before looking for the object.
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    obj = _first_json_object(candidate)
    if obj is None:
        # Some runs put the object before the separator, or omit it entirely.
        obj = _first_json_object(text)
    return obj


def _first_json_object(text: str) -> Optional[dict]:
    """Return the first balanced ``{...}`` in ``text`` that parses as JSON."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = text[start:i + 1]
                    try:
                        data = json.loads(chunk)
                    except ValueError:
                        break
                    if isinstance(data, dict):
                        return data
                    break
        start = text.find("{", start + 1)
    return None


class PromptRewriter:
    """Loads the rewriter checkpoint on first use and expands a prompt."""

    def __init__(self, model_id: str = PE_T2I_MODEL_ID, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.system_prompt: Optional[str] = None
        self.is_loaded = False

    # ------------------------------------------------------------------ state
    @property
    def available(self) -> bool:
        """Whether runnable weights are on disk, without loading them."""
        return _local_snapshot(self.model_id, require_weights=True) is not None

    def _require(self):
        from modules.runtime import model_access

        model_access.require(self.model_id)

    # ------------------------------------------------------------------ load
    def load(self) -> bool:
        """Load tokenizer and model from the local cache.

        Returns False rather than raising when the checkpoint is absent, so the
        caller can carry on with the original prompt.
        """
        if self.is_loaded:
            return True

        snap = _local_snapshot(self.model_id, require_weights=True)
        if snap is None:
            logger.warning(
                f"prompt rewriter {self.model_id} is not on disk; generating "
                f"from the prompt as written. Fetch it with: "
                f"python scripts/download_models.py --model {self.model_id}")
            return False

        self._require()

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading prompt rewriter {self.model_id} from {snap}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(snap), local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                str(snap), dtype=torch.bfloat16,
                device_map="auto", local_files_only=True).eval()
            prompt_path = snap / "system_prompt.txt"
            self.system_prompt = (
                prompt_path.read_text().strip() if prompt_path.is_file() else "")
            if not self.system_prompt:
                logger.warning(
                    "system_prompt.txt missing; the rewriter needs its own "
                    "system prompt to produce the documented output shape")
            self.is_loaded = True
            return True
        except Exception as exc:
            logger.warning(f"prompt rewriter failed to load: {exc}")
            self.model = self.tokenizer = None
            self.is_loaded = False
            return False

    # ---------------------------------------------------------------- rewrite
    def rewrite(self, prompt: str) -> Optional[Rewrite]:
        """Expand ``prompt`` into a detailed request plus a recommended size.

        Returns None whenever the rewrite cannot be trusted, so the caller keeps
        the prompt the user actually typed.
        """
        if not prompt or not prompt.strip():
            return None
        if not self.load():
            return None

        import torch

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.model.generate(**inputs, **GENERATE_KWARGS)
            decoded = self.tokenizer.decode(
                out[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)
        except Exception as exc:
            logger.warning(f"prompt rewriter run failed, using prompt as "
                           f"written: {type(exc).__name__}: {exc}")
            return None

        data = parse_rewriter_output(decoded)
        if not data:
            logger.warning(
                "prompt rewriter returned no parsable JSON; using the prompt "
                "as written")
            return None

        new_prompt = str(data.get("rewritten_prompt") or "").strip()
        if not new_prompt:
            logger.warning("prompt rewriter returned an empty rewritten_prompt")
            return None

        ratio = str(data.get("wh_ratio") or "").strip() or None
        width, height = WH_RATIO_TO_SIZE.get(ratio, DEFAULT_SIZE)
        thinking, _, _ = decoded.partition(THINK_CLOSE)
        return Rewrite(prompt=new_prompt, wh_ratio=ratio,
                       width=width, height=height,
                       thinking=thinking.strip(), raw=data)

    # ---------------------------------------------------------------- release
    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        self.system_prompt = None
        self.is_loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def rewriter_enabled(config: dict = None) -> bool:
    """Whether the rewriter should run for this launch.

    Off unless asked for, because it is a second 9B model on the GPU and an
    unrequested 17.55 GiB is a poor surprise. ``EC_QI21_PROMPT_REWRITER`` is the
    switch for a single run; the ``prompt_rewriter`` key in the model config is
    the durable one.
    """
    from modules.backends.image_21 import _env_flag

    env = os.environ.get("EC_QI21_PROMPT_REWRITER")
    if env is not None:
        return _env_flag("EC_QI21_PROMPT_REWRITER")
    if config and "prompt_rewriter" in config:
        value = config.get("prompt_rewriter")
        if isinstance(value, str):
            return value.strip().casefold() in ("1", "true", "yes", "on")
        return bool(value)
    return False
