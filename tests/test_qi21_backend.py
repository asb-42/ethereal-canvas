"""Behavioural checks for the Qwen-Image 2.1 backend against the real library."""
import os, pathlib, sys, types

#: Located from this file, so the suite runs wherever the repo happens to sit.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

#: An isolated acceptance store: these checks must neither clear nor depend on
#: the operator's real acceptance of the upstream terms.
_TMP_STORE = str(pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
                  / f"qi21-acceptances-{os.getpid()}.json")
pathlib.Path(_TMP_STORE).unlink(missing_ok=True)
os.environ["EC_MODEL_LICENSE_FILE"] = _TMP_STORE
os.environ.pop("EC_MODEL_LICENSE_ACCEPTED", None)

from PIL import Image
import diffusers
from modules.runtime import model_access, paths
from modules.runtime.paths import is_qwen_image_21

FAILS = []
def check(label, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {label}" + (f"  [{detail}]" if detail else ""))
    if not ok: FAILS.append(label)

rec = model_access.ACCEPTANCE_FILE
assert rec == pathlib.Path(_TMP_STORE), (
    f"refusing to run: the suite would clear {rec}, which holds real acceptances")
rec.unlink(missing_ok=True)

# ------------------------------------------------------- 1. the gate still holds
from modules.backends.image_21 import QwenImage21Backend
try:
    QwenImage21Backend().load()
    check("gate: refuses to fetch with no acceptance", False, "it proceeded")
except model_access.ModelLicenseNotAccepted:
    check("gate: refuses to fetch with no acceptance", True)
except Exception as exc:
    check("gate: refuses to fetch with no acceptance", False, f"{type(exc).__name__}: {exc}")

# ------------------------------------------------------- a faithful stand-in
CAPTURED = {}
class Fake:
    def __init__(self, **_kw): self.calls = 0
    def to(self, device): return self
    def enable_attention_slicing(self): CAPTURED["sliced"] = True
    def enable_model_cpu_offload(self, **kw): CAPTURED["offload"] = kw
    def __call__(self, *, prompt=None, image=None, negative_prompt=None,
                 true_cfg_scale=1.0, height=None, width=None,
                 num_inference_steps=40, num_images_per_prompt=1,
                 generator=None, output_type="pil", output_resolution=1024,
                 use_kv_cache=True, **rest):
        CAPTURED["kw"] = dict(prompt=prompt, image=image,
                              negative_prompt=negative_prompt,
                              true_cfg_scale=true_cfg_scale, height=height,
                              width=width, num_inference_steps=num_inference_steps,
                              num_images_per_prompt=num_images_per_prompt,
                              generator=generator, output_type=output_type,
                              output_resolution=output_resolution,
                              use_kv_cache=use_kv_cache, **rest)
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

class FakeClass:
    @classmethod
    def from_pretrained(cls, name=None, **kw):
        CAPTURED['fetched'] = (name, kw)
        return Fake()
setattr(diffusers, "QwenImage21Pipeline", FakeClass)

os.environ["EC_MODEL_LICENSE_ACCEPTED"] = "1"
b = QwenImage21Backend(); b.load()
check("load: attention slicing engaged", CAPTURED.get("sliced") is True)
check("load: no set_progress_bar on a class that lacks it", True)

# ------------------------------------------------------- 2. text-to-image call
out = b.generate_image("a red fox", seed=None)
kw = CAPTURED["kw"]
check("t2i: returns images", len(out) == 1)
check("t2i: 40 steps (card value)", kw["num_inference_steps"] == 40, str(kw["num_inference_steps"]))
check("t2i: true_cfg_scale 1.0 (flow matching)", kw["true_cfg_scale"] == 1.0)
check("t2i: no guidance_scale leaked in", "guidance_scale" not in kw)
check("t2i: no mask argument invented", not [k for k in kw if "mask" in k])
check("t2i: no rgba flag fabricated", "rgba" not in kw)

b.generate_image("a red fox", negative_prompt="blurry", width=1024, height=768,
                 num_steps=10, num_images_per_prompt=3)
kw = CAPTURED["kw"]
check("t2i: negative prompt forwarded", kw["negative_prompt"] == "blurry")
check("t2i: explicit size honoured", (kw["width"], kw["height"]) == (1024, 768))
check("t2i: step override honoured", kw["num_inference_steps"] == 10)
check("t2i: batch count honoured", kw["num_images_per_prompt"] == 3)

# ------------------------------------------------------- 3. seeding/determinism
b.generate_image("a red fox", seed=1234)
kw = CAPTURED["kw"]
check("seed: generator supplied", kw["generator"] is not None)
check("seed: kv cache pinned off for reproducibility", kw["use_kv_cache"] is False)
os.environ["EC_QI21_KV_CACHE"] = "1"
b.generate_image("a red fox", seed=1234)
check("seed: cache opt-back-in is explicit", CAPTURED["kw"]["use_kv_cache"] is True)
del os.environ["EC_QI21_KV_CACHE"]
b.generate_image("a red fox")
check("no seed: no generator, cache untouched", CAPTURED["kw"].get("generator") is None)

# ------------------------------------------------------- 4. editing / references
img = Image.new("RGB", (64, 64), "red")
b.edit_images([img], "make it blue")
kw = CAPTURED["kw"]
check("edit: single reference passed as one image", isinstance(kw["image"], Image.Image))
check("edit: uses the card's 2048 edit resolution", kw["output_resolution"] == 2048,
      str(kw["output_resolution"]))
b.edit_images([img] * 4, "combine", reference_images=[img] * 2)
check("edit: several references become a list",
      isinstance(CAPTURED["kw"]["image"], list) and len(CAPTURED["kw"]["image"]) == 6)
b.edit_images([img], "x", strength=0.3)
check("edit: strength accepted then ignored, not sent", "strength" not in CAPTURED["kw"])
for label, fn in (
    ("edit: no source image is refused", lambda: b.edit_images([], "x")),
    ("edit: >10 references refused", lambda: b.edit_images([img] * 11, "x")),
    ("t2i: ambiguous multi-prompt refused", lambda: b.generate_image(["a", "b"])),
):
    try: fn(); check(label, False, "no error")
    except ValueError: check(label, True)
    except Exception as e: check(label, False, f"{type(e).__name__}")

# ------------------------------------------------------- 5. optional offload
os.environ["EC_QI21_SEQUENTIAL_CPU_OFFLOAD"] = "1"
QwenImage21Backend().load()
check("offload: sequential component order applied",
      CAPTURED.get("offload", {}).get("model_cpu_offload_seq") ==
      "text_encoder->transformer->vae", str(CAPTURED.get("offload")))
del os.environ["EC_QI21_SEQUENTIAL_CPU_OFFLOAD"]

# ------------------------------------------------------- 6. routing + memory
from modules.backends.adapter import BackendAdapter
a21 = BackendAdapter({"generate_model": paths.QWEN_QI21_MODEL_ID,
                      "edit_model": paths.QWEN_QI21_MODEL_ID})
legacy = BackendAdapter({})
check("route: 2.1 name selects the unified backend",
      type(a21._t2i_backend(paths.QWEN_QI21_MODEL_ID)).__name__ == "QwenImage21Backend")
check("route: edit role shares that backend",
      type(a21._edit_backend(paths.QWEN_QI21_MODEL_ID)).__name__ == "QwenImage21Backend")
check("route: legacy t2i name unchanged",
      type(legacy._t2i_backend("Qwen/Qwen-Image-2512")).__name__ == "TextToImageBackend")
check("route: legacy edit name unchanged",
      type(legacy._edit_backend("Qwen/Qwen-Image-Edit-2511")).__name__ == "ImageEditBackend")
check("route: detector ignores the 2512 name", is_qwen_image_21("Qwen/Qwen-Image-2512") is False)

from modules.memory.manager import MemoryManager
mm = MemoryManager()
est = lambda m: mm.estimate_required_memory(m)["fp16_full"]
n21 = est(paths.QWEN_QI21_MODEL_ID)
old = est("Qwen/Qwen-Image-Edit-2511")
check("memory: 2.1 gets its own estimate", n21 != old, f"2.1={n21:.0f}MB legacy={old:.0f}MB")
check("memory: 2.1 estimate is inside the published 44 GiB", n21 < 44 * 1024)
check("memory: 2.1 estimate is not the legacy table copied", n21 > old)
check("paths: unified cache dir", paths.QWEN_QI21_CACHE.name == "Qwen-Image-2.1")

b.release()
check("release: pipeline dropped", b.pipeline is None and b.is_loaded is False)


def test_qi21_backend():
    """The checks above execute on import; this asserts their outcome."""
    assert not FAILS, "; ".join(FAILS)


if __name__ == "__main__":
    for _f in FAILS:
        print(f"  failing: {_f}")
    sys.exit(1 if FAILS else 0)
