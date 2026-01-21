# Ethereal Canvas — Design Rationale & Non-Goals

## 1. Purpose

This document records the design intent behind Ethereal Canvas.

It exists to:
- explain architectural decisions
- make tradeoffs explicit
- prevent accidental scope expansion
- preserve long-term maintainability

This document is normative.

---

## 2. Core Design Principles

### 2.1 Determinism First

Ethereal Canvas prioritizes:
- reproducibility
- auditability
- controlled randomness

This is why:
- seeds are mandatory
- global randomness is centrally managed
- all executions are logged and committed

---

### 2.2 Unix Philosophy

The system is intentionally composed of:
- small modules
- narrow responsibilities
- explicit contracts

This:
- improves debuggability
- reduces cognitive load
- mitigates large-context limitations
- simplifies replacement of components

---

### 2.3 Explicit Over Implicit

The system avoids:
- hidden state
- automatic retries
- magic defaults

Every important decision:
- is logged
- is reproducible
- can be reasoned about

---

### 2.4 Human-Readable Artifacts

Logs, configuration, and documentation are:
- text-based
- human-readable
- version-controlled

Git is treated as a first-class audit system.

---

## 3. Technology Choices — Rationale

### 3.1 Gradio UI

**Chosen because:**
- fast iteration
- minimal boilerplate
- browser-based access
- good Pinokio compatibility

**Rejected alternatives:**
- heavy frontend frameworks
- custom JS stacks

---

### 3.2 Pinokio Deployment

**Chosen because:**
- one-click installation
- reproducible environments
- controlled lifecycle
- local-first philosophy

Pinokio defines the operational boundary.

---

### 3.3 Transformers as Primary Inference Stack

**Chosen because:**
- model diversity
- unified APIs
- strong ecosystem
- compatibility with large VRAM GPUs

Diffusers support is deferred but anticipated.

---

### 3.4 Git-Based Logging

**Chosen because:**
- immutability
- diffability
- human inspection
- natural rollback support

Databases were intentionally avoided.

---

## 4. Architectural Tradeoffs

### 4.1 Single-User, Single-Task Execution

**Benefits**
- simplicity
- determinism
- reduced failure modes

**Cost**
- no concurrency
- lower throughput

This is an explicit and acceptable tradeoff.

---

### 4.2 No Authentication or Access Control

**Benefits**
- reduced complexity
- fewer dependencies
- simpler deployment

**Cost**
- unsuitable for untrusted networks

Security is delegated to the environment.

---

### 4.3 No Background Workers

**Benefits**
- predictable execution
- simpler debugging
- clear control flow

**Cost**
- blocking operations

This aligns with the deterministic design goal.

---

## 5. Non-Goals

Ethereal Canvas explicitly does NOT aim to:

- be a multi-user system
- be cloud-native
- provide an API-first interface
- perform task scheduling
- manage user accounts
- optimize throughput over reproducibility
- hide model limitations
- act as a general-purpose image editor

Any feature pushing in these directions
requires explicit architectural review.

---

## 6. Deferred Capabilities

The following are intentionally deferred:

- native inpainting
- diffusers-based pipelines
- model ensembles
- advanced UI workflows
- plugin systems
- distributed inference

Deferred does not mean rejected.

---

## 7. Future Evolution Guardrails

Future changes must preserve:

- backend abstraction boundaries
- determinism guarantees
- audit trail completeness
- small-module philosophy

Violations require explicit documentation.

---

## 8. Authority

This document defines the authoritative design intent
of Ethereal Canvas.

If behavior contradicts this document,
the behavior is considered a defect.