# Ethereal Canvas — Formal Architecture Specification

## 1. Purpose

Ethereal Canvas is a modular, network-accessible image generation and
image editing system designed for reproducibility, auditability, and
long-term extensibility.

The system prioritizes:
- Deterministic execution
- Clear separation of concerns
- Replaceable model backends
- Human-readable operational logs
- Minimal coupling between components

This document defines the authoritative architecture of the system.

---

## 2. Architectural Principles

### 2.1 Unix Philosophy

Each component:
- Has a single, well-defined responsibility
- Communicates via explicit data structures
- Can be debugged and reasoned about independently

No component relies on implicit global state beyond configuration files.

### 2.2 Determinism First

Given identical:
- input image(s)
- prompt text
- seed
- model backend and version

The system must produce identical outputs, within the limits of
underlying model implementation.

### 2.3 Model Agnosticism

The system does not assume:
- a specific image model
- a specific inference framework

All model-specific logic is isolated behind a strict adapter interface.

---

## 3. High-Level System Overview

The system consists of the following layers:

1. User Interface Layer (Gradio)
2. Orchestration Layer (Job Runner)
3. Domain Logic Layer (Prompts, Images)
4. Model Backend Layer
5. Persistence & Audit Layer (Logs, Git)

Each layer may depend only on layers below it.

---

## 4. Module Responsibilities

### 4.1 img_read

**Responsibility**
- Validate and load image files
- Normalize images into internal representation

**Invariants**
- Only JPEG and PNG are accepted
- Image metadata must be preserved when possible

---

### 4.2 img_write

**Responsibility**
- Persist images to disk
- Embed reproducibility metadata

**Required Metadata**
- Prompt text
- Seed
- Model identifier
- Timestamp

---

### 4.3 prompt_engine

**Responsibility**
- Normalize prompt text
- Manage seeds
- Enforce prompt validity
- Establish reproducibility context

The prompt engine is the sole authority for seed generation.

---

### 4.4 model_adapter

**Responsibility**
- Define the canonical interface for image models

**Required Capabilities**
- Text → Image
- Image → Image
- (Optional) Inpainting

No UI or orchestration logic may bypass this interface.

---

### 4.5 Backend Implementations

#### Qwen Image Backend
- Uses HuggingFace transformers
- Optimized for high-VRAM GPUs
- No native inpainting (as of current version)

#### Diffusers Backend
- Uses HuggingFace diffusers
- Provides native inpainting
- Serves as reference backend for image editing tasks

---

### 4.6 job_runner

**Responsibility**
- Orchestrate task execution
- Apply seed control
- Select backend
- Coordinate logging and persistence

The job runner is stateless between tasks except for session persistence.

---

### 4.7 ui_gradio

**Responsibility**
- Provide a browser-based user interface
- Collect user inputs
- Display outputs and logs

The UI layer must never directly interact with model backends.

---

### 4.8 logging

**Responsibility**
- Maintain a human-readable Markdown audit trail
- Commit logs to Git after each task
- Record system fingerprint and task metadata

Logs are append-only.

---

## 5. Configuration Model

All mutable behavior must be configurable via YAML files.

Hard-coded values are prohibited except where required by third-party APIs.

---

## 6. Extension Points

The architecture explicitly supports:
- Additional model backends
- Additional task types (e.g., inpainting)
- Alternative user interfaces
- External orchestration layers

No extension may require changes to unrelated modules.

---

## 7. Non-Goals

The following are explicitly out of scope:
- Multi-user concurrency
- Distributed inference
- Cloud-native orchestration
- Closed-source model integration

---

## 8. Stability Guarantees

The interfaces defined in this document are considered stable.
Breaking changes require:
- explicit documentation
- version increment
- migration notes

---

## 9. Authority

This document is the authoritative architectural reference
for Ethereal Canvas.

In case of conflict between implementation and this document,
this document takes precedence.