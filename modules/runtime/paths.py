"""
Runtime path utilities for Ethereal Canvas.

Provides centralized path management for all runtime artifacts.
Enforces Opencode runtime hygiene standards.
"""

from pathlib import Path
from datetime import datetime
import os

# -------------------------------------------------
# Canonical runtime root
#
# Anchored to the repository root rather than the working directory. A relative
# ``Path("runtime")`` made every cache path below resolve against whatever
# directory the process happened to start in, so the downloader and the loader
# only agreed when both were launched from the repo root. scripts/run.sh does
# cd there, but scripts/download_models.py is run as a file from anywhere, and
# the two then wrote and read different trees.
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(os.environ.get("EC_RUNTIME_ROOT") or REPO_ROOT / "runtime")

# -------------------------------------------------
# Runtime directories
# -------------------------------------------------
LOGS_DIR = RUNTIME_ROOT / "logs"
OUTPUTS_DIR = RUNTIME_ROOT / "outputs"
CACHE_DIR = RUNTIME_ROOT / "cache"
TMP_DIR = RUNTIME_ROOT / "tmp"

# -------------------------------------------------
# Model cache directories
# -------------------------------------------------
MODEL_CACHE_DIR = REPO_ROOT / "models"
QWEN_T2I_CACHE = MODEL_CACHE_DIR / "Qwen-Image-2512"
QWEN_I2I_CACHE = MODEL_CACHE_DIR / "Qwen-Image-Edit-2511"
# The 2.1 release is one checkpoint for both roles, so it gets a cache entry of
# its own rather than being squeezed under either legacy name.
QWEN_QI21_CACHE = MODEL_CACHE_DIR / "Qwen-Image-2.1"
QWEN_QI21_MODEL_ID = "Qwen/Qwen-Image-2.1"

def is_qwen_image_21(model_name: str) -> bool:
    """Whether this name is the unified Qwen-Image 2.1 release."""
    return bool(model_name) and "Qwen-Image-2.1" in model_name


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def ensure_runtime_dirs() -> None:
    """Create all runtime directories if they do not exist."""
    for path in (LOGS_DIR, OUTPUTS_DIR, CACHE_DIR, TMP_DIR):
        path.mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
    """Generate consistent timestamp for filenames."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def output_image_path(prefix: str, suffix: str = "png") -> Path:
    """Generate unique output image path."""
    ensure_runtime_dirs()
    return OUTPUTS_DIR / f"{prefix}_{timestamp()}.{suffix}"

def output_edit_path(prefix: str, suffix: str = "png") -> Path:
    """Generate unique edit image path."""
    ensure_runtime_dirs()
    return OUTPUTS_DIR / f"{prefix}_{timestamp()}.{suffix}"

def output_inpaint_path(prefix: str, suffix: str = "png") -> Path:
    """Generate unique inpaint image path."""
    ensure_runtime_dirs()
    return OUTPUTS_DIR / f"{prefix}_{timestamp()}.{suffix}"

def log_file_path(name: str, suffix: str = "log") -> Path:
    """Generate log file path."""
    ensure_runtime_dirs()
    return LOGS_DIR / f"{name}_{timestamp()}.{suffix}"

def tmp_path(name: str = None, suffix: str = "") -> Path:
    """Generate temporary file path."""
    ensure_runtime_dirs()
    if name and suffix:
        return TMP_DIR / f"{name}_{timestamp()}.{suffix}"
    elif name:
        return TMP_DIR / f"{name}_{timestamp()}"
    else:
        return TMP_DIR / timestamp()

def model_cache_path(model_name: str) -> Path:
    """Generate model cache path."""
    ensure_runtime_dirs()
    return MODEL_CACHE_DIR / model_name


def model_cache_dir_for(model_name: str) -> Path:
    """Where the weights for ``model_name`` belong on disk.

    Single source of truth for the download side and the load side. Both used
    to derive the location independently, which is how the 2.1 backend ended up
    reading from the ambient hub cache while the downloader wrote elsewhere.
    Accepts the bare directory name as well as the ``Org/Name`` form.
    """
    from modules.runtime.model_access import normalize_model_id

    key = normalize_model_id(model_name)
    tail = key.split("/", 1)[1] if "/" in key else key
    return MODEL_CACHE_DIR / tail


def config_path() -> Path:
    """The model configuration file to read.

    Defaults to the committed ``config/model_config.yaml``. ``EC_MODEL_CONFIG``
    points the application at a different file, which is how an operator runs the
    2.1 release without editing the committed default that every other user gets.
    """
    override = os.environ.get("EC_MODEL_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    return REPO_ROOT / "config" / "model_config.yaml"


def load_model_config() -> dict:
    """Read the model configuration, or say plainly why it could not be read.

    The three call sites each opened a relative ``config/model_config.yaml``
    inside a bare ``except:``, falling back to the 2512/2511 pair on any error at
    all. A typo in an override, a malformed file, or running from another
    directory therefore did not fail: it quietly served the legacy pair, so a
    run meant to prove the 2.1 backend could report on the legacy one without a
    word of complaint. An explicit override that cannot be read is now fatal.
    """
    import yaml

    path = config_path()
    overridden = bool(os.environ.get("EC_MODEL_CONFIG"))
    if not path.is_file():
        if overridden:
            raise FileNotFoundError(
                f"EC_MODEL_CONFIG points at {path}, which does not exist")
        raise FileNotFoundError(
            f"no model configuration at {path}; run the application from the "
            f"repository root or set EC_MODEL_CONFIG")

    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        raise ValueError(f"unreadable model configuration {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"model configuration {path} did not parse to a mapping "
            f"(got {type(data).__name__})")
    return data

def cleanup_temp_files() -> None:
    """Clean all temporary files older than 1 hour."""
    if not TMP_DIR.exists():
        return
    
    import time
    current_time = time.time()
    one_hour_ago = current_time - 3600
    
    try:
        for file_path in TMP_DIR.iterdir():
            if file_path.is_file():
                file_mtime = file_path.stat().st_mtime
                if file_mtime < one_hour_ago:
                    file_path.unlink()
                    print(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        print(f"Error cleaning temp files: {e}")

def get_cache_usage() -> dict:
    """Get cache directory usage statistics."""
    cache_stats = {}
    for cache_name in ["Qwen-Image-2512", "Qwen-Image-Edit-2511"]:
        cache_path = model_cache_path(cache_name)
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            cache_stats[cache_name] = {
                "size_bytes": total_size,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": len(list(cache_path.rglob("*")))
            }
        else:
            cache_stats[cache_name] = {"size_bytes": 0, "size_mb": 0, "file_count": 0}
    
    return cache_stats