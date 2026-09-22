# Ethereal Canvas - Code Audit Report

**Date:** 2026-02-23  
**Branch:** v3.0-audit-fixes  
**Auditor:** Automated Code Review

---

## Summary

Comprehensive audit of the Ethereal Canvas codebase identified **6 critical bugs** and **3 stub implementations**. All critical bugs have been fixed in this branch.

---

## Critical Bugs Fixed

### 1. Undefined `kwargs` in text_to_image.py (Lines 342, 350)
**Severity:** Critical  
**Issue:** `**kwargs` used without defining `kwargs` dictionary  
**Fix:** Replaced with explicit `load_kwargs` dictionary

### 2. Invalid Pipeline Import in image_inpaint.py (Lines 53, 116, 149)
**Severity:** Critical  
**Issue:** `QwenImageEditPlusPipeline` does not exist in diffusers  
**Fix:** Replaced with `DiffusionPipeline` generic import

### 3. torch.autocast() with None torch (image_inpaint.py:197)
**Severity:** Critical  
**Issue:** `torch.autocast()` called when `torch` could be `None`  
**Fix:** Added proper null-check and used `torch.inference_mode()` instead

### 4. Missing metadata parameter in writer.py (Line 16)
**Severity:** High  
**Issue:** `write_image()` required `metadata` but was called without it  
**Fix:** Made `metadata` optional, added support for both `ImageData` and `PIL.Image`

### 5. Duplicate Backend Initialization in ui.py (Lines 269-281)
**Severity:** Medium  
**Issue:** Backend adapter initialized twice in `stop_status_updates()`  
**Fix:** Removed duplicate initialization code

### 6. Incorrect Indentation in text_to_image.py (Line 438)
**Severity:** Low  
**Issue:** Comment before `if monitor:` had incorrect indentation  
**Fix:** Corrected indentation

---

## Stub Implementations Identified

| Location | Type | Status |
|----------|------|--------|
| `text_to_image.py::_generate_stub()` | Fallback | Acceptable for development |
| `image_edit.py::edit()` | Stub when pipeline fails | Acceptable as fallback |
| `image_inpaint.py` | Completely disabled | Needs implementation |

---

## Architecture Compliance

### ✅ Compliant
- Module separation follows architecture_spec.md
- Backend adapter interface matches model_backend_spec.md
- Runtime paths follow operations_spec.md

### ⚠️ Needs Attention
- Inpainting backend is disabled (not per spec)
- Git-based audit trail not fully implemented in UI

---

## Recommendations

1. **Enable Inpainting Backend** - Currently using edit backend as fallback
2. **Add Input Validation** - Prompt validation exists but is not enforced
3. **Implement Git Audit Trail** - Per operations_spec.md, logs should be committed after each task
4. **Add Type Hints** - Some functions lack complete type annotations

---

## Files Modified

```
modules/backends/text_to_image.py     - Bug fixes
modules/backends/image_inpaint.py     - Bug fixes
modules/img_write/writer.py           - API improvement
modules/ui_gradio/ui.py               - Removed duplicate code
.gitignore                            - Added venv/, *.bak
```

---

## Testing Recommendations

Before merging:
1. Run `python -m py_compile modules/backends/*.py`
2. Test T2I generation with stub mode
3. Test I2I editing with stub mode
4. Verify UI loads without errors
