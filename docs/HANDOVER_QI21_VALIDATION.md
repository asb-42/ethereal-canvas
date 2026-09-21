# Handover: validate Qwen-Image 2.1 on the large-VRAM machine

Audience: a fresh agent session with no memory of the session that produced this.
Read this file fully before running anything. Everything below was established
by inspecting the code and library on the previous machine; the items marked
**UNVERIFIED** are exactly what you are here to settle.

## 1. Mission

The Qwen-Image 2.1 integration is **written and structurally verified but has
never executed**. The previous machine had 23.5 GiB VRAM against a ~44 GiB
checkpoint, so no real load or generation has ever happened. You are on a
128 GiB box: your job is to run it, gather evidence, calibrate the two guessed
tables, and report. You are not here to add features.

Two rules outrank the rest:

1. **Do not break the legacy path.** `.venv` (diffusers 0.36.0, transformers
   4.57.6, torch 2.5.1+cu121) serves the working 2512/2511 pair. Never upgrade
   it in place. The 2.1 stack lives in `.venv-qi21` beside it.
2. **Do not break the license gate.** See section 5.

## 2. Repo state you should start from

Branch `v3.0-audit-fixes`, fast-forwarded with `main`. Confirm before you begin:

```bash
git rev-parse --short HEAD        # expect 9aa04a8 or newer
git status --short                # expect empty
```

Relevant commits: `9aa04a8` (2.1 backend), `58bf21c` (dependency pin),
`fc177bc` (license gate), `3774f0f` (AGPL licence).

## 3. Build the environment, then verify it cheaply before expensively

```bash
./scripts/setup_qi21_env.sh
```

This creates `.venv-qi21`, writes a venv-level `pip.conf`, installs
`requirements_qi21.txt` (a full freeze, diffusers pinned to commit
`80c7ed262aeffbeb43ef13ae04baeb9b84515a69`), and **fails loudly unless
`QwenImage21Pipeline` imports**. A green setup script is your first gate.

Then, in this order, stopping at the first failure:

```bash
# a) the interpreter and the class
.venv-qi21/bin/python -c "import diffusers, torch; \
print(diffusers.__version__, torch.__version__, torch.cuda.is_available(), \
torch.cuda.get_device_properties(0).total_memory // 2**30)"

# b) the ported behavioural suite (fake pipeline, no GPU work, ~33 checks)
.venv-qi21/bin/python tests/test_qi21_backend.py; echo "exit=$?"

# c) legacy regression: the old stack must still serve the UI
PORT=7860 ./scripts/run.sh   # then, from another shell:
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/gradio_api/info
```

(b) must print `ok` lines and exit `0`. (c) must give `200` twice. If either
fails, **stop and report** - do not proceed to model loading.

Note `tests/test_qi21_backend.py` uses an isolated acceptance store
(`EC_MODEL_LICENSE_FILE`) so it can test refusal without touching your real
acceptance record. Do not remove that.

## 4. Hardware pre-flight (do this first, on the new box)

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
df -h .                                          # need >= ~60 G free for weights
.venv-qi21/bin/python -c "import torch; print(torch.cuda.device_count())"
.venv-qi21/bin/python -c "import flash_attn"     # if present, confirm it imports
```

`torch` must see every card, and the total memory must actually read ~128 G.
If `import diffusers` raises a flash-attn `RuntimeError` about an `aten` schema
mismatch, the interpreter is wrong, not the library - see the trap in section 6.

## 5. Weights, and the gate in front of them

The app **refuses to fetch weights until the operator has accepted the upstream
terms**, and it does so before any network call. That is intended behaviour,
not a bug to route around:

```bash
.venv-qi21/bin/python scripts/accept_model_license.py --list
.venv-qi21/bin/python scripts/accept_model_license.py \
    --model Qwen/Qwen-Image-2.1 --yes
```

- Acceptances land in `runtime/model_license_acceptances.json`, which is
  gitignored. **Never commit it**, and never copy third-party licence text into
  the repository - the gate stores only the licence name and the upstream URL.
- `EC_MODEL_LICENSE_ACCEPTED=1` exists for headless runs but deliberately
  cannot unlock a model with no recorded terms. Prefer the explicit `--yes`.
- Fetch the weights only after the acceptance above:

```bash
.venv-qi21/bin/python scripts/download_models.py --model Qwen-Image-2.1
```

Expect roughly 44 GiB into `models/Qwen-Image-2.1/`. Confirm the on-disk size
afterwards and report it against the published figure.

## 6. Traps that cost real time on the previous machine

- **Tool output mangles identifiers.** Repeated underscores in dunders are
  collapsed in display, and package/class names are sometimes rewritten
  (`diffusers`, `transformers`, `PIL`, `from_pretrained`). **Never copy an
  identifier out of displayed output.** Confirm names inside the sandbox
  (`hasattr`, `inspect.signature`, real calls) before acting on them.
- **`grep` intermittently returns nothing for content that is there.** When a
  grep looks suspiciously empty, re-check with a Python substring search.
- **Never `source .venv*/bin/activate`.** On the previous box a stale
  `VIRTUAL_ENV` in `pyvenv.cfg` made `python` resolve to an unrelated system
  interpreter, producing a flash-attn `RuntimeError` inside `import diffusers`
  that looks like a library fault. `scripts/run.sh` binds the interpreter by
  path for this reason; override with `EC_PYTHON`.
- **pip may be configured with an unreachable NVIDIA index**
  (`pypi.ngc.nvidia.com`) in the user-level config, which stalls installs. The
  venv-level `pip.conf` written by the setup script overrides it. If installs
  stall, run `python -m pip config list` and report it rather than editing
  global config.
- **`pytest` is not installed in either virtualenv.** The files under `tests/`
  are pytest-style, so run them directly: `python tests/test_x.py`. Installing
  pytest is a dependency change - flag it, don't just add it.
- **`docs/AUDIT_REPORT.md` is not trustworthy.** At least one of its "fixes"
  replaced the real `QwenImageEditPlusPipeline` with a base `DiffusionPipeline`.
  Verify any claim against code before acting.
- **Do not add mask support to 2.1.** The installed `QwenImage21Pipeline`
  accepts no `mask`/`mask_image` parameter, which is why the inpaint tab stays
  on the 2511 checkpoint behind an `_accepts_mask()` gate that raises rather
  than silently discarding a mask. Card images showing circles/annotations are
  ComfyUI/vLLM/SGLang side features, not diffusers ones.

## 7. The actual validation, cheapest to dearestive

Select the model by config, not code - the adapter routes on the model name:

```yaml
generate_model: "Qwen/Qwen-Image-2.1"
edit_model:     "Qwen/Qwen-Image-2.1"
```

`config/model_config.yaml` documents this block. **Leave the committed default
on the 2512/2511 pair** unless you are told otherwise: flipping a default for
everyone is not yours to decide.

Then, recording evidence at every step:

| # | Run | What to capture |
|---|---|---|
| 1 | T2I at 512 and 768 | peak VRAM, wall time, that images are returned |
| 2 | T2I at 1024 | same |
| 3 | T2I at 2048 (card's recommended) | same; note quality and any OOM |
| 4 | I2I single reference, `output_resolution=2048` | peak VRAM; output size |
| 5 | Multi-reference: 2, 5 and 10 images | does >1 behave as "all images apply to every prompt" |
| 6 | Same seed, twice | outputs must be byte-identical (`use_kv_cache` is pinned off when seeded) |
| 7 | `strength=0.3` on an edit | must be logged as ignored, not applied |
| 8 | RGBA request | see the open question below |
| 9 | `negative_prompt` set vs unset | confirm it actually changes output |

Both `nvidia-smi` lines above were run here and are the accepted flag forms
(`--format=csv`, `-l 1`); the `noheader.units` format modifiers are rejected on
this driver. A torch-side alternative that also works is
`torch.cuda.mem_get_info()`, which returns used and total for the whole device.

Sample during the run:

```bash
nvidia-smi -l 1 --query-gpu=memory.used --format=csv > peaks.log   # -l, not --l
```

Report **peak**, not final, and report the command line you used.

## 8. Two tables that are currently guesses

- `modules/memory/manager.py` - `QI21_BASE_REQUIREMENTS` is marked
  **UNCALIBRATED** and derived from the published 44 GiB footprint. Replace it
  with your measured peaks. For context, the legacy table claims 16 GB for a
  ~58 GB checkpoint, which shows how these drift.
- `modules/backends/image_21.py` - `RECOMMENDED_EDIT_RESOLUTION`,
  `NUM_INFERENCE_STEPS`, `TRUE_CFG_SCALE` were taken from the model card and
  checked against the installed signature, **not** against rendered output.
  If a real run contradicts the card, say so and show both.

## 9. Open questions only a real run can answer

1. **RGBA.** The card advertises it; the installed call surface exposes only
   `output_type='pil'`. No `rgba` flag was invented in code. Find out how RGBA
   actually emerges and report the mechanism before anything is added.
2. `true_cfg_scale`: card says keep 1.0 for flow matching. Does deviating
   visibly degrade output, as claimed?
3. Does `output_resolution` behave as the only size control when `height`/`width`
   are left `None`?
4. Are the `prompt_embeds` + image padding-mask incompatibility reported
   upstream still present in this pinned commit? They surface as a
   `RuntimeError` by design.
5. Does `EC_QI21_SEQUENTIAL_CPU_OFFLOAD=1` (text encoder, transformer, VAE)
   actually help at 128 GiB, or is it dead weight here?

## 10. Report back in this shape

```
environment:  diffusers <ver> @ <commit>, transformers <ver>, torch <ver>, GPU <name>/<VRAM>
setup script: pass/fail, plus the class-import line it prints
suite:        tests/test_qi21_backend.py -> exit <n>, <k> checks
legacy boot:  / -> <code>, /gradio_api/info -> <code>, log errors: <n>
weights:      <bytes> on disk vs published 47,357,320,470
runs 1-9:     one line each: command -> peak VRAM, wall time, outcome
calibration:  proposed QI21_BASE_REQUIREMENTS from measured peaks
divergences:  anywhere reality disagreed with this document
unverified:   what you did not manage to test, and why
```

Do not report "works" without the line that proves it.

## 11. Guardrails

- One focused commit, message style matching `git log` (imperative subject,
  `-` bullets explaining *why*). Push the branch; ask before touching `main`.
- No new dependencies without flagging it. No in-place upgrade of `.venv`.
- No weights, no `runtime/`, no `__pycache__` in commits - check
  `git status --short` immediately before committing.
- If a change grows beyond this document's scope, stop and ask.
