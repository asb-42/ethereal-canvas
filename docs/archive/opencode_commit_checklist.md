# Opencode Commit Checklist

Mandatory verification checklist for all Ethereal Canvas commits.

## Checklist Items

Before committing, Opencode MUST verify:

### 🚫 PROHIBITED - These must NEVER be committed
- [ ] No files in `__pycache__/` or `*.pyc` or `*.pyo` are staged
- [ ] No virtual environment directories (`.venv/`, `venv/`, `env/`) are staged  
- [ ] No runtime artifacts (`runtime/`, `models/`, `outputs/`) are staged
- [ ] No model files or HuggingFace caches are staged
- [ ] Git status is clean except for intended source changes
- [ ] All new files are either:
  - source code in `modules/`, `scripts/`, `config/`
  - documentation in `docs/`
  - configuration files
- [ ] Commit message follows project conventions

### ✅ REQUIRED - These MUST be present
- [x] Runtime directories exist and are ignored by `.gitignore`
- [x] All new code paths go through `modules/runtime/paths.py`
- [x] All backends inherit from `modules/models/base.py`
- [x] No hardcoded paths outside runtime utilities
- [x] All file writes use runtime paths via `output_image_path()` etc.
- [x] Metadata is captured and attached to all outputs
- [x] Seeds are explicit parameters in generation calls
- [x] No hidden randomness or implicit defaults

### 🔧 TECHNICAL - These verify implementation quality
- [x] No global mutable state in backends
- [x] All imports are at the top of files
- [x] No circular dependencies
- [x] Exception handling preserves context
- [x] Resource cleanup is in `finally:` blocks
- [x] Types are properly hinted
- [x] No live modification of sys.path at import time
- [x] Deterministic behavior where specified

### 📋 DOCUMENTATION - These must be updated when features change
- [x] README.md reflects current capabilities
- [x] User guide explains runtime behavior
- [x] API documentation matches implementation
- [x] Architecture decisions are recorded in appropriate `.md` files

### 🔄 TESTING - Optional but recommended for major changes
- [ ] New functionality covered by tests
- [ ] Edge cases are handled
- [ ] Performance is not degraded
- [ ] Memory usage stays within expected bounds

## Usage

Before each commit:
1. Run this checklist
2. Fix any failures
3. Commit only when ALL required items are checked

## Validation Commands

```bash
# Check for prohibited files
python -c "
import subprocess
result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True)
prohibited = [line for line in result.stdout.split('\n') 
            if any(item in line for item in ['__pycache__', '.venv/', 'models/', 'runtime/']) 
            for item in ['*.pyc', '*.pyo']]
print('PROHIBITED FILES IN STAGING:', bool(prohibited))
"

# Verify gitignore is working
python -c "
import subprocess
result = subprocess.run(['git', 'check-ignore'], capture_output=True)
print('GITIGNORE STATUS:', result.returncode == 0)
"
```

---

**Remember**: If any REQUIRED item fails, the commit MUST be blocked and the issue fixed.