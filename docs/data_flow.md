# Ethereal Canvas — Data & Control Flow Specification

## 1. Scope

This document defines how data and control flow through Ethereal Canvas
during normal operation.

It describes:
- task lifecycles
- data transformations
- control handoff between modules
- logging and persistence boundaries
- error propagation rules

This document is normative.

---

## 2. High-Level Execution Flow

At a high level, execution follows this sequence:

1. User submits input via UI
2. UI invokes job runner
3. Job runner normalizes inputs
4. Model backend performs inference
5. Output is written to disk
6. Logs are updated and committed
7. UI presents results

No step is skipped.

---

## 3. Control Flow by Layer

### 3.1 UI Layer (Gradio)

**Inputs**
- Prompt text
- Optional seed
- Optional input image
- Optional mask image (future)

**Responsibilities**
- Validate presence of required fields
- Forward user intent to job runner
- Display status, logs, and outputs

**Control Rules**
- The UI never calls model backends directly
- The UI never writes files directly
- All execution is delegated to job runner

---

### 3.2 Job Runner Layer

The job runner is the central control authority.

**Responsibilities**
- Determine task type (generate / edit / inpaint)
- Normalize and validate inputs
- Establish reproducibility context
- Select backend
- Coordinate execution
- Handle failures
- Trigger logging and persistence

**Control Flow**
1. Receive task request
2. Normalize prompt
3. Resolve seed
4. Set global seed state
5. Load input image(s) if required
6. Dispatch task to backend
7. Receive ImageData result
8. Write output image
9. Emit structured log entry
10. Return output reference

The job runner is synchronous by design.

---

### 3.3 Prompt Lifecycle

**Stages**
1. Raw prompt input (string)
2. Normalization (whitespace, validation)
3. Seed association
4. Parameter binding
5. Metadata embedding

**Rules**
- Prompt normalization happens exactly once
- The normalized prompt object is immutable
- All downstream components receive the same prompt object

---

### 3.4 Seed & Determinism Flow

**Seed Sources**
- Explicit user-provided seed
- Generated seed (if none provided)

**Propagation**
- The resolved seed is:
  - applied globally (random, numpy, torch)
  - embedded into metadata
  - logged
  - committed to Git

**Invariant**
Given identical inputs and environment,
the same seed must produce identical outputs.

---

### 3.5 Image Input Flow

**Applicable Tasks**
- edit
- inpaint

**Stages**
1. File path validation
2. Image decoding
3. Metadata extraction
4. Conversion to internal ImageData

**Rules**
- Only JPEG and PNG are accepted
- Invalid images abort execution immediately
- ImageData is immutable once created

---

### 3.6 Model Backend Invocation

**Invocation Contract**
- Job runner calls backend methods via model_adapter interface only
- Backend receives:
  - ImageData (if applicable)
  - PromptObject

**Backend Responsibilities**
- Perform inference
- Return ImageData
- Not write files
- Not perform logging

Backends must be stateless across tasks.

---

### 3.7 Output Flow

**Stages**
1. ImageData returned by backend
2. Metadata enrichment
3. File path resolution
4. Image serialization
5. Disk write

**Output Naming**
- Deterministic naming based on:
  - task type
  - seed
  - timestamp (optional)

---

### 3.8 Logging & Audit Flow

**Log Events**
- Application startup
- Task start
- Task completion
- Errors
- Shutdown

**Log Format**
- Markdown
- Append-only
- Structured fields

**Git Integration**
- Each log write is followed by:
  - git add
  - git commit

Logs are the authoritative execution history.

---

## 4. Error Propagation Rules

### 4.1 Error Sources

Errors may originate from:
- Invalid user input
- Missing files
- Model backend failures
- Resource exhaustion
- Serialization errors

---

### 4.2 Propagation Policy

**Rules**
- Errors propagate upward immediately
- No silent retries
- No implicit recovery

**Responsibilities**
- Backend: raise explicit exceptions
- Job runner: log error and abort task
- UI: display error message

Partial outputs must not be written.

---

## 5. Session Persistence Flow

**Session Data**
- Last prompt
- Last seed
- Last output path

**Lifecycle**
- Saved after each successful task
- Loaded at application startup

Session data is advisory and must not affect determinism.

---

## 6. Shutdown Flow

**Triggers**
- SIGINT
- SIGTERM
- Controlled UI shutdown

**Steps**
1. Stop accepting new tasks
2. Complete current task (if possible)
3. Flush logs
4. Release GPU memory
5. Exit process

---

## 7. Concurrency Model

- Single-user
- Single-task at a time
- No parallel execution

This is an explicit design choice.

---

## 8. Guarantees

The system guarantees:
- Ordered execution
- Complete audit trail
- Deterministic behavior
- No hidden state

---

## 9. Authority

This document defines the canonical data and control flow
for Ethereal Canvas.

Any deviation must be documented explicitly.