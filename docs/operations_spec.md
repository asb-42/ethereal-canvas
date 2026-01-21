# Ethereal Canvas — Operational & Deployment Specification

## 1. Purpose

This document defines the operational lifecycle of Ethereal Canvas.

It specifies:
- installation and deployment
- runtime assumptions
- configuration management
- failure handling
- upgrade and rollback strategy
- operational safeguards

This document is normative.

---

## 2. Deployment Model

### 2.1 Primary Deployment Target

The primary deployment target is:
- Linux
- Local machine or workstation
- GPU-equipped system (optional but recommended)
- Managed via Pinokio

No cloud infrastructure is assumed.

---

## 2.2 Network Model

- The application runs as a local server process
- The UI is accessed via a web browser
- The listening port is configurable
- Intended usage is:
  - local machine
  - LAN
  - VPN

No authentication is enforced by default.

---

## 3. Installation Lifecycle

### 3.1 Installation Artifacts

The following artifacts define installation:

- `pinokio.json`
- `scripts/install.sh`
- `requirements.txt`
- `config/*.yaml`

Pinokio is the authoritative orchestrator.

---

### 3.2 Installation Steps

1. Pinokio invokes `scripts/install.sh`
2. Python virtual environment is created
3. Dependencies are installed from `requirements.txt`
4. CUDA availability is detected
5. Model weights are downloaded
6. Installation completes without starting the app

Installation must be idempotent.

---

### 3.3 Environment Isolation

- A dedicated virtual environment is used
- No global Python packages are modified
- All runtime dependencies are contained within the project directory

---

## 4. Runtime Lifecycle

### 4.1 Startup

On startup, the application performs:

1. Configuration loading
2. Environment validation
3. Backend initialization
4. Model loading
5. System fingerprint logging
6. UI server launch

Startup must fail fast if any critical step fails.

---

### 4.2 Runtime Operation

During normal operation:

- Only one task is processed at a time
- Tasks are synchronous
- GPU memory is reclaimed after each task
- Logs are written continuously

The application maintains no hidden background state.

---

### 4.3 Shutdown

Shutdown may be triggered by:
- User action
- SIGINT / SIGTERM
- Pinokio lifecycle events

Shutdown sequence:
1. Stop accepting new tasks
2. Complete current task if possible
3. Flush logs
4. Release model and GPU resources
5. Exit process cleanly

---

## 5. Configuration Management

### 5.1 Configuration Files

All mutable behavior is controlled via YAML files:

- `server_config.yaml`
- `model_config.yaml`
- Optional future configs

Configuration files:
- are read at startup
- are not modified at runtime

---

### 5.2 Configuration Changes

Configuration changes require:
- application restart
- no code changes

Hot-reloading is explicitly out of scope.

---

## 6. Logging & Audit Operations

### 6.1 Log Storage

- Logs are stored in `logs/runlog.md`
- Logs are append-only
- Logs are human-readable

---

### 6.2 Git-Based Audit Trail

After each task:
- logs are staged
- logs are committed to Git

The Git repository is the authoritative execution history.

---

## 7. Failure Modes

### 7.1 Expected Failures

- Invalid user input
- Missing or corrupt image files
- Out-of-memory errors
- Model inference failures

---

### 7.2 Failure Handling Policy

- Fail fast
- Log all failures
- Do not retry automatically
- Do not produce partial outputs

---

### 7.3 Recovery

Recovery actions:
- Restart application
- Adjust configuration
- Re-run task

No automatic state recovery is attempted.

---

## 8. Upgrade Strategy

### 8.1 Code Upgrades

- Performed via Git pull or Pinokio update
- Require application restart

---

### 8.2 Dependency Upgrades

- Performed by updating `requirements.txt`
- Require re-running `install.sh`

---

### 8.3 Model Upgrades

- Change model identifier in configuration
- Trigger fresh model download
- Previous outputs remain valid and auditable

---

## 9. Rollback Strategy

Rollback is performed by:
- checking out a previous Git commit
- restarting the application

Because outputs are immutable and logged, rollback is safe.

---

## 10. Resource Assumptions

### 10.1 Minimum Requirements

- Python 3.10+
- 16 GB system RAM
- Disk space sufficient for models and outputs

---

### 10.2 Recommended Configuration

- NVIDIA GPU with ≥24 GB VRAM
- 64–128 GB system RAM
- Fast local SSD

---

## 11. Security Considerations

- No authentication by default
- No encryption by default
- Intended for trusted environments only

Hardening is the responsibility of the operator.

---

## 12. Authority

This document defines the authoritative operational model
for Ethereal Canvas.

In case of conflict, this document takes precedence.