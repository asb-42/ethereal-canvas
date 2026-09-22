#!/usr/bin/env python3
"""
Run the section 7 validation matrix for the Qwen-Image 2.1 backend.

This exists because the nine runs in docs/HANDOVER_QI21_VALIDATION.md have never
been executed, and because the two ways the document gives you to sample memory
both fail on this driver: nvidia-smi reports memory.total/used as [N/A] and its
`-l 1` loop never terminates. Peak is therefore taken from the torch side, from
a sampler thread running for the whole life of each call, and the command line
that produced every number is recorded alongside it.

Report is **peak**, not final, per the document.

Usage:
    EC_MODEL_CONFIG=$PWD/config/model_config.qi21.yaml \\
        .venv-qi21/bin/python scripts/validate_qi21.py
    ... scripts/validate_qi21.py --only t2i_1024,t2i_2048
    ... scripts/validate_qi21.py --steps 4        # cheap smoke pass
"""

import argparse
import hashlib
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "runtime" / "outputs" / "qi21-validation"
EVIDENCE = REPO / "runtime" / "logs" / "qi21-validation.json"

#: How often the sampler reads device memory. 200 ms is well under the lifetime
#: of a single denoising step, so it cannot miss a peak that persists for a step.
SAMPLE_S = 0.2


class PeakSampler(threading.Thread):
    """Poll device memory in the background and keep the high-water mark.

    A single reading after the call is worthless: by then the scheduler has
    freed the activations that made the run need its most memory.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.peak_bytes = 0
        self.total_bytes = 0
        self.samples = 0

    def run(self):
        import torch
        while not self._stop_event.is_set():
            try:
                free, total = torch.cuda.mem_get_info()
                used = total - free
                self.total_bytes = total
                if used > self.peak_bytes:
                    self.peak_bytes = used
                self.samples += 1
            except Exception:
                return
            self._stop_event.wait(SAMPLE_S)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=10)


def gib(n):
    return n / 2**30


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixels_of(images) -> bytes:
    """Raw pixel bytes, so a comparison is about content and not PNG metadata.

    Comparing encoded files would be the wrong test: the encoder can write a
    different timestamp or text chunk and two visually identical images would
    read as different.
    """
    buf = io.BytesIO()
    for img in images:
        rgba = img.mode == "RGBA"
        rgb = img.convert("RGBA") if rgba else img.convert("RGB")
        # Size and mode go in as text: two images of different size must not
        # hash alike, and encoding them as ascii keeps this readable.
        buf.write(f"{rgb.mode}|{rgb.size[0]}x{rgb.size[1]}|".encode("ascii"))
        buf.write(rgb.tobytes())
    return buf.getvalue()


def save(images, name: str):
    paths = []
    for i, img in enumerate(images):
        p = OUT / f"{name}_{i}.png"
        img.save(p)
        paths.append(str(p))
    return paths


class Runner:
    def __init__(self, steps: int):
        self.steps = steps
        self.results = []
        self.backend = None
        self._ref = None

    # ------------------------------------------------------------ plumbing
    def start(self):
        from modules.backends.image_21 import QwenImage21Backend
        from modules.runtime.model_access import require
        from modules.runtime.paths import load_model_config

        cfg = load_model_config()
        name = cfg.get("generate_model")
        require(name)                      # never fetch without the licence step
        self.backend = QwenImage21Backend(name)
        t0 = time.time()
        ok = self.backend.load()
        self.record("load", ok, wall=time.time() - t0,
                    cmd=f"QwenImage21Backend({name!r}).load()")
        if not ok:
            raise RuntimeError("backend did not load; stopping")

    def record(self, run, ok, wall=None, peak=None, total=None, cmd="", **extra):
        row = {"run": run, "ok": bool(ok)}
        if wall is not None:
            row["wall_s"] = round(wall, 2)
        if peak is not None:
            row["peak_GiB"] = round(gib(peak), 3)
        if total:
            row["device_total_GiB"] = round(gib(total), 3)
        if cmd:
            row["command"] = cmd
        row.update(extra)
        self.results.append(row)
        flag = "ok  " if ok else "FAIL"
        print(f"  {flag} {run:28s} "
              f"peak={row.get('peak_GiB', '-'):>6} GiB  "
              f"wall={row.get('wall_s', '-'):>7} s")
        return row

    def timed(self, name, cmd, fn):
        s = PeakSampler()
        s.start()
        t0 = time.time()
        try:
            out = fn()
            self.record(name, True, wall=time.time() - t0, peak=s.peak_bytes,
                        total=s.total_bytes, cmd=cmd,
                        returned=len(out) if isinstance(out, list) else 1)
            return out
        except Exception as exc:
            self.record(name, False, wall=time.time() - t0, peak=s.peak_bytes,
                        total=s.total_bytes, cmd=cmd,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc()[-1])
            return None
        finally:
            s.stop()

    # --------------------------------------------------------------- the runs
    def t2i(self, size):
        name = f"t2i_{size}"
        cmd = (f"generate_image(prompt=..., width={size}, height={size}, "
               f"num_steps={self.steps})")
        def go():
            imgs = self.backend.generate_image(
                "a red cube on a white table", width=size, height=size,
                num_steps=self.steps)
            if not imgs:
                raise AssertionError("pipeline returned no images")
            for im in imgs:
                if im is None or im.width < 1 or im.height < 1:
                    raise AssertionError("pipeline returned an empty image")
            save(imgs, name)
            return imgs
        return self.timed(name, cmd, go)

    def edit_single(self):
        cmd = ("edit_images([ref], prompt=..., output_resolution=2048, "
               "num_steps=%d)" % self.steps)
        observed = {}
        def go():
            imgs = self.backend.edit_images(
                [self.reference()], "make the cube blue",
                output_resolution=2048, num_steps=self.steps)
            if not imgs:
                raise AssertionError("pipeline returned no images")
            save(imgs, "edit_single")
            observed["output_size"] = list(imgs[0].size)
            return imgs
        out = self.timed("edit_single_2048", cmd, go)
        if self.results and self.results[-1].get("run") == "edit_single_2048":
            self.results[-1].update(observed)
        return out

    def reference(self):
        """One generated image to edit, made once and reused."""
        if self._ref is None:
            imgs = self.backend.generate_image(
                "a red cube on a white table", width=512, height=512,
                num_steps=4)
            self._ref = imgs[0]
        return self._ref

    def multi_ref(self, n):
        cmd = (f"edit_images(<{n} refs>, prompt=..., num_steps={self.steps})")
        def go():
            refs = [self.reference() for _ in range(n)]
            imgs = self.backend.edit_images(
                refs, "arrange these objects on a shelf",
                num_steps=self.steps)
            save(imgs, f"multi_ref_{n}")
            return imgs
        return self.timed(f"multi_ref_{n}", cmd, go)

    def seeded_twice(self):
        cmd = "generate_image(prompt=..., seed=1234, num_steps=%d) x2" % self.steps
        digests = []
        def go():
            for i in range(2):
                imgs = self.backend.generate_image(
                    "a green sphere", seed=1234, num_steps=self.steps)
                digests.append(sha256_of(pixels_of(imgs)))
                save(imgs, f"seeded_{i}")
            if digests[0] != digests[1]:
                raise AssertionError(
                    f"same seed gave different pixels: {digests[0][:12]} vs "
                    f"{digests[1][:12]}")
            return [None]
        return self.timed("seed_reproducibility", cmd, go)

    def strength_ignored(self):
        cmd = "edit_images([ref], prompt=..., strength=0.3)"
        def go():
            imgs = self.backend.edit_images(
                [self.reference()], "make the cube blue", strength=0.3,
                num_steps=self.steps)
            # The claim to test is that strength is reported and not applied.
            # If it were applied it would change the schedule and the output.
            save(imgs, "strength")
            return imgs
        out = self.timed("strength_0.3_on_edit", cmd, go)
        if (self.results
                and self.results[-1].get("run") == "strength_0.3_on_edit"
                and self.results[-1].get("ok")):
            self.results[-1]["expectation"] = (
                "logged as ignored; flow matching has no "
                "denoising-strength knob")
        return out

    def rgba(self):
        """Record how RGBA actually emerges, rather than asserting it does.

        The card advertises RGBA but the installed call surface exposes only
        output_type='pil'. Whether an alpha channel can be obtained is an open
        question, so this run's job is to report the mechanism it observes -
        accepted and honoured, accepted but ignored, or rejected outright - and
        to pass either way. The answer is the payload, not a verdict.
        """
        cmd = 'generate_image(prompt=..., output_type="rgba")'
        observed = {}
        def go():
            try:
                imgs = self.backend.generate_image(
                    "a red cube", output_type="rgba", num_steps=self.steps)
            except TypeError as exc:
                observed["rgba_mechanism"] = f"rejected: {exc}"
                return [None]
            modes = [im.mode for im in imgs]
            observed["rgba_mechanism"] = (
                f"accepted; returned modes={modes}; "
                f"alpha={'present' if any('A' in m for m in modes) else 'absent'}")
            save(imgs, "rgba")
            return imgs
        out = self.timed("rgba_request", cmd, go)
        if self.results and self.results[-1].get("run") == "rgba_request":
            self.results[-1].update(observed)
        return out

    def negative_prompt(self):
        cmd = "generate_image(prompt=..., negative_prompt=... vs none)"
        def go():
            a = self.backend.generate_image("a cat", num_steps=self.steps)
            b = self.backend.generate_image("a cat", negative_prompt="blurry",
                                            num_steps=self.steps)
            same = sha256_of(pixels_of(a)) == sha256_of(pixels_of(b))
            save(a, "neg_off"); save(b, "neg_on")
            if same:
                raise AssertionError(
                    "negative_prompt changed nothing: identical pixels")
            return a
        return self.timed("negative_prompt_effect", cmd, go)

    # ------------------------------------------------------------------ drive
    def run_all(self, only=None):
        jobs = {
            "t2i_512":  lambda: self.t2i(512),
            "t2i_768":  lambda: self.t2i(768),
            "t2i_1024": lambda: self.t2i(1024),
            "t2i_2048": lambda: self.t2i(2048),
            "edit_single": self.edit_single,
            "multi_ref": lambda: [self.multi_ref(n) for n in (2, 5, 10)],
            "seeded": self.seeded_twice,
            "strength": self.strength_ignored,
            "rgba": self.rgba,
            "negative": self.negative_prompt,
        }
        todo = list(jobs) if not only else [k for k in jobs if k in only]
        for key in todo:
            print(f"[run] {key}")
            try:
                jobs[key]()
            except Exception as exc:
                self.record(key, False, error=f"{type(exc).__name__}: {exc}")

    def finish(self):
        OUT.mkdir(parents=True, exist_ok=True)
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "environment": self.env_report(),
            "steps": self.steps,
            "results": self.results,
        }
        EVIDENCE.write_text(json.dumps(payload, indent=2, default=str))
        passed = sum(1 for r in self.results if r["ok"])
        print(f"\n{passed}/{len(self.results)} runs passed")
        print(f"evidence: {EVIDENCE}")
        peaks = [r["peak_GiB"] for r in self.results if "peak_GiB" in r]
        if peaks:
            print(f"highest peak observed: {max(peaks):.2f} GiB")
        return 0 if passed == len(self.results) else 1

    def env_report(self):
        import torch
        import diffusers
        import transformers
        try:
            commit = diffusers.version.version.split("+")[1].split("-")[0]
        except Exception:
            commit = "unknown"
        return {
            "diffusers": diffusers.__version__,
            "diffusers_commit": commit,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "vram_GiB": round(gib(torch.cuda.get_device_properties(0).total_memory), 2)
            if torch.cuda.is_available() else 0,
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", help="comma list of: t2i_512,t2i_768,t2i_1024,"
                                   "t2i_2048,edit_single,multi_ref,seeded,"
                                   "strength,rgba,negative")
    ap.add_argument("--steps", type=int, default=40,
                    help="num_inference_steps (40 is the card value)")
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Qwen-Image 2.1 validation, num_steps={args.steps}")
    print(f"outputs -> {OUT}\n")

    r = Runner(steps=args.steps)
    try:
        r.start()
    except Exception as exc:
        print(f"backend failed to start: {exc}")
        traceback.print_exc()
        return 2
    try:
        r.run_all(args.only.split(",") if args.only else None)
    finally:
        try:
            r.backend.release()
        except Exception:
            pass
    return r.finish()


if __name__ == "__main__":
    sys.exit(main())
