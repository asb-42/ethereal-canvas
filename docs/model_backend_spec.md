# Ethereal Canvas — Model Backend & Abstraction Specification

## 1. Purpose

This document defines the formal contract between Ethereal Canvas
and all image model backends.

Its purpose is to:
- guarantee interchangeability of backends
- enforce determinism and auditability
- prevent architectural erosion
- allow independent evolution of models and UI

This document is normative.

---

## 2. Conceptual Model

A model backend is a stateless service object that:
- receives normalized inputs
- performs inference
- returns image outputs
- does not manage orchestration, UI, or persistence

All model-specific logic is isolated inside the backend.

---

## 3. Required Backend Interface

Every backend MUST implement the following interface.

### 3.1 Lifecycle Methods

load() -> None
shutdown() -> None

**load**
- Initializes model weights
- Allocates required resources
- Must be idempotent

**shutdown**
- Releases all allocated resources
- Clears GPU memory if applicable
- Leaves no residual state

---

### 3.2 Generation Methods

generate_image(prompt: PromptObject) -> ImageData
edit_image(image: ImageData, prompt: PromptObject) -> ImageData

**generate_image**
- Produces an image from text only

**edit_image**
- Produces an image conditioned on an input image

If native image editing is not supported, `edit_image` MAY internally
delegate to `generate_image`, but this limitation must be documented.

---

### 3.3 Optional Inpainting Method

inpaint(image: ImageData, mask: ImageData, prompt: PromptObject) -> ImageData

If not implemented, backend MUST raise `NotImplementedError`.

---

## 4. Input Contracts

### 4.1 PromptObject

Backends receive a fully normalized PromptObject.

**Backend Assumptions**
- Prompt text is valid
- Seed has already been resolved
- Parameters are immutable

Backends MUST NOT modify the prompt object.

---

### 4.2 ImageData

Backends receive ImageData objects.

**Invariants**
- Pixel data is valid
- Dimensions are consistent
- Metadata may be present but must not be relied upon

Backends MUST treat ImageData as immutable.

---

## 5. Determinism Contract

### 5.1 Required Behavior

Given:
- identical PromptObject
- identical ImageData (if applicable)
- identical backend version
- identical runtime environment

The backend MUST produce identical output.

---

### 5.2 Seed Handling

Backends MUST NOT:
- generate their own seeds
- override global seed state
- introduce hidden randomness

All randomness must be derived from the externally supplied seed.

---

## 6. Output Contract

### 6.1 ImageData Output

Backends MUST return ImageData objects containing:
- pixel data
- correct dimensions
- valid format

Backends MUST NOT:
- write files to disk
- embed metadata
- perform logging

---

## 7. Prohibited Responsibilities

A backend MUST NOT:
- access the filesystem (except model loading)
- perform Git operations
- emit logs
- interact with the UI
- manage configuration files
- spawn background threads

Violation of these rules is considered an architectural error.

---

## 8. Backend Classes

### 8.1 Transformers Backend

**Characteristics**
- Uses HuggingFace transformers
- Supports large foundational models
- Optimized for high-VRAM GPUs

**Strengths**
- High image fidelity
- Unified text+vision models

**Limitations**
- Limited native image editing
- Inpainting support model-dependent

---

### 8.2 Diffusers Backend

**Characteristics**
- Uses HuggingFace diffusers
- Pipeline-based architecture
- Explicit support for editing and inpainting

**Strengths**
- Mature image editing workflows
- Native inpainting

**Limitations**
- Model fragmentation
- Higher configuration complexity

---

## 9. Backend Selection

Backend selection is determined exclusively via configuration.

No runtime auto-switching is permitted.

---

## 10. Adding a New Backend

To add a new backend, the following steps are required:

1. Implement the full adapter interface
2. Document backend capabilities and limitations
3. Add configuration support
4. Add at least one deterministic test
5. Update documentation

No other module may require changes.

---

## 11. Compatibility & Versioning

Backend interfaces are considered stable.

Breaking changes require:
- new backend version
- updated documentation
- explicit migration notes

---

## 12. Authority

This document defines the authoritative backend abstraction
for Ethereal Canvas.

In case of conflict, this document takes precedence.