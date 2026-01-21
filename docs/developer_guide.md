# Developer Guide

## Architecture Principles

Ethereal Canvas follows Unix philosophy:
- Small, focused modules
- Single responsibility principle
- Explicit interfaces
- Composable components

## How to Add a New Backend

1. **Implement ModelAdapter Interface**
   - Inherit from `modules/model_adapter/adapter.py:ModelAdapter`
   - Implement all abstract methods: `load()`, `generate_image()`, `edit_image()`, `inpaint()`, `shutdown()`

2. **Backend Module Structure**
   ```
   modules/my_backend/
   ├── __init__.py
   ├── loader.py          # Main backend implementation
   ├── memory.py          # Optional: memory management utilities
   └── config.py          # Optional: configuration helpers
   ```

3. **Register Backend**
   - Update `config/model_config.yaml` with backend configuration
   - Add backend selection logic in job runner if needed

## How to Add a New UI Feature

1. **Gradio Interface**
   - Edit `modules/ui_gradio/interface.py`
   - Add components in the appropriate layout
   - Wire up event handlers

2. **Backend Integration**
   - Add corresponding methods to job runner
   - Update logging to capture new operations

## Debugging Tips

### Common Issues

**CUDA Out of Memory**
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

**Import Errors**
```bash
# Check environment
source .venv/bin/activate
pip list | grep torch
```

**Port Conflicts**
```bash
# Find what's using the port
lsof -i:7860
```

### Logging

All operations are logged in `logs/runlog.md` with:
- Timestamp
- Task details
- System fingerprint
- Git commit tracking

Use structured logging in `modules/logging/logger.py`

## Dependency Management

All Python dependencies are declared in `requirements.txt`.

Rules:
- No Python package may be installed ad-hoc in scripts
- Version pinning should be added only when required
- CUDA-specific handling is performed in `install.sh`, not in requirements.txt

This ensures reproducible and auditable environments.

## Known Limitations

1. **Inpainting**: Interface defined but not implemented
2. **Model Loading**: Only Qwen-Image-2512 supported currently
3. **Batch Generation**: Single image generation per request
4. **Memory Usage**: Large models require significant GPU memory

## Testing

Run tests with:
```bash
python -m pytest tests/
```

Focus areas:
- Reproducibility (same seed = same output)
- Error handling (invalid inputs, missing files)
- Integration (UI → backend → output)

## Contributing

1. Follow existing code patterns
2. Add comprehensive logging
3. Update documentation
4. Include tests for new features
5. Commit messages must be English and descriptive