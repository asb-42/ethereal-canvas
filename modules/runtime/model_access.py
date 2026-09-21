"""
Model access control.

Ethereal Canvas ships no model weights. Every checkpoint the app can load is
distributed separately by its own upstream, under that upstream's own terms,
and those terms are not the same as this application's terms.

This module is the single gate in front of every code path that can pull a
checkpoint, whether that is a pipeline ``from_pretrained()`` call or the
standalone download script. Nothing may reach the network for weights without
an explicit, recorded acceptance from the person running the application.

Deliberate design choices:

* The license text is never copied into this repository. We reference the name
  and the upstream URL, and the user reads the terms at the source. That keeps
  this repository from redistributing someone else's licensed document and
  keeps the reference honest when upstream changes it.
* Acceptance is recorded per model, locally, in ``runtime/`` which is ignored
  by git. An acceptance is therefore never committed and never travels with a
  clone: each person who runs the app accepts for themselves.
* A model we cannot identify is treated as needing review, not as permitted.
* This module performs no downloads and no imports of heavy dependencies, so
  it is safe to call from a CLI entry point before torch is loaded.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import os

# -------------------------------------------------
# Refusal
# -------------------------------------------------

class ModelLicenseNotAccepted(RuntimeError):
    """Raised when a model is requested without a recorded license acceptance."""


# -------------------------------------------------
# Known upstream terms
#
# `license` is the name of the agreement as published by the upstream, and
# `url` points at it. Nothing here reproduces the text of any agreement.
# -------------------------------------------------

MODEL_LICENSES = {
    "Qwen/Qwen-Image-2.1": {
        "license": "Qwen Research License Agreement",
        "url": "https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
    },
    "Qwen/Qwen-Image-2512": {
        # Verified from the upstream model card (cardData.license = apache-2.0,
        # gated = false) rather than assumed.
        "license": "Apache License 2.0",
        "url": "https://huggingface.co/Qwen/Qwen-Image-2512",
    },
    "Qwen/Qwen-Image-Edit-2511": {
        # Verified from the upstream model card (cardData.license = apache-2.0,
        # gated = false) rather than assumed.
        "license": "Apache License 2.0",
        "url": "https://huggingface.co/Qwen/Qwen-Image-Edit-2511",
    },
}

# Set to "1" for headless or CI runs where the operator has already read the
# terms. It is read per model, so it does not let an unknown model through.
ENV_ACCEPTED = "EC_MODEL_LICENSE_ACCEPTED"

#: Where acceptances are recorded.
#:
#: Anchored to the repository root, not the working directory: with a relative
#: path an operator could accept the terms while sitting in one directory and
#: still be refused when launching the app from another, which reads as the gate
#: ignoring them. EC_MODEL_LICENSE_FILE relocates it, so tests and CI can use a
#: throwaway store instead of the operator's real one.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_LICENSE_FILE = "EC_MODEL_LICENSE_FILE"
ACCEPTANCE_FILE = Path(os.environ.get(ENV_LICENSE_FILE)
                       or REPO_ROOT / "runtime" / "model_license_acceptances.json")


# -------------------------------------------------
# Identity
# -------------------------------------------------

def normalize_model_id(model_id: str) -> str:
    """Match the many spellings a model arrives under onto one table key."""
    if not model_id:
        return ""
    raw = str(model_id).strip().strip("/")
    lowered = raw.casefold()
    for known in MODEL_LICENSES:
        known_lowered = known.casefold()
        if lowered == known_lowered:
            return known
        # A bare "Qwen-Image-2.1", a cache directory name, or a local path
        # ending in the repo name all mean the same checkpoint.
        if lowered.endswith("/" + known.split("/", 1)[1].casefold()):
            return known
    return raw


def terms_for(model_id: str) -> dict:
    """Return the recorded terms for a model, or a review-required entry."""
    key = normalize_model_id(model_id)
    entry = MODEL_LICENSES.get(key)
    if entry is None:
        return {
            "model_id": model_id,
            "known": False,
            "license": None,
            "url": None,
            "requires_review": True,
        }
    return {
        "model_id": key,
        "known": True,
        "license": entry["license"],
        "url": entry["url"],
        "requires_review": entry["license"] is None,
    }


# -------------------------------------------------
# Acceptance record
# -------------------------------------------------

def _read_record() -> dict:
    if not ACCEPTANCE_FILE.is_file():
        return {}
    try:
        data = json.loads(ACCEPTANCE_FILE.read_text())
    except (ValueError, OSError):
        # A damaged record must never silently grant access.
        return {}
    return data if isinstance(data, dict) else {}


def is_accepted(model_id: str) -> bool:
    """Whether this operator has accepted this model's terms on this machine."""
    if os.environ.get(ENV_ACCEPTED) == "1":
        # The blanket switch still requires that we know the terms at all.
        return terms_for(model_id)["known"]
    entry = _read_record().get(normalize_model_id(model_id))
    return bool(entry and entry.get("accepted_at"))


def accept(model_id: str, accepted_by: str = "user") -> dict:
    """Record an explicit acceptance. Call this only from an action the
    operator took on purpose, never on a load path."""
    info = terms_for(model_id)
    if info["requires_review"]:
        raise ModelLicenseNotAccepted(
            f"Cannot record an acceptance for {model_id!r}: this checkout has no verified "
            f"license record for it. Read the terms upstream, add them to MODEL_LICENSES, "
            f"then accept. Reference: {info['url']}"
        )
    record = _read_record()
    entry = {
        "license": info["license"],
        "url": info["url"],
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accepted_by": accepted_by,
    }
    record[info["model_id"]] = entry
    ACCEPTANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCEPTANCE_FILE.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return entry


# -------------------------------------------------
# The gate
# -------------------------------------------------

def describe(model_id: str) -> str:
    """Human-readable statement of what is being fetched and under what."""
    info = terms_for(model_id)
    license_label = info["license"] or "terms not yet recorded in this checkout"
    url = info["url"] or "upstream model card"
    return (
        f"{info['model_id']} is provided by its upstream under the {license_label}.\n"
        f"     Read it before downloading: {url}\n"
        f"     This application neither contains nor redistributes those weights."
    )


def require(model_id: str) -> None:
    """Gate every weight-fetching path. Raises unless acceptance is recorded."""
    if is_accepted(model_id):
        return
    info = terms_for(model_id)
    raise ModelLicenseNotAccepted(
        f"Refusing to fetch {model_id!r}: the license acceptance for this model is not "
        f"recorded on this machine.\n\n"
        f"  {describe(model_id)}\n\n"
        f"  To proceed, read the terms above and then record your own acceptance:\n"
        f"      python scripts/accept_model_license.py --model {info['model_id']!r} --yes\n"
        f"  For an unattended run where you have already read them, set "
        f"{ENV_ACCEPTED}=1."
    )
