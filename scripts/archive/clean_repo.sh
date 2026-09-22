#!/usr/bin/env bash
set -e

# =============================================================================
# Ethereal Canvas: Repository Hygiene Script
# Moves logs, outputs, and models into runtime/, cleans git, updates .gitignore
# =============================================================================

echo "[Hygiene] Starting repository cleanup..."

# 1. Create runtime subdirectories
mkdir -p runtime/logs runtime/outputs runtime/models runtime/cache

# 2. Move known runtime files into runtime/
echo "[Hygiene] Moving logs..."
if [ -d "logs" ]; then
    mv logs/* runtime/logs/ 2>/dev/null || true
fi

echo "[Hygiene] Moving outputs..."
if [ -d "outputs" ]; then
    mv outputs/* runtime/outputs/ 2>/dev/null || true
fi

echo "[Hygiene] Moving models..."
if [ -d "models" ]; then
    mv models/* runtime/models/ 2>/dev/null || true
fi

# 3. Update .gitignore
echo "[Hygiene] Updating .gitignore..."
cat > .gitignore <<EOL
# Runtime directories
runtime/logs/
runtime/outputs/
runtime/models/
runtime/cache/

# Python artifacts
__pycache__/
*.pyc
.venv/

# OS artifacts
.DS_Store
EOL

git add .gitignore

# 4. Remove runtime files from git index if already tracked
echo "[Hygiene] Cleaning git index for runtime files..."
git rm -r --cached runtime/logs 2>/dev/null || true
git rm -r --cached runtime/outputs 2>/dev/null || true
git rm -r --cached runtime/models 2>/dev/null || true

# 5. Show status before commit
echo "[Hygiene] Current git status (short):"
git status --short

# 6. Suggest commit message
echo "[Hygiene] Ready to commit core code, configs, docs, and scripts. Suggested message:"
echo "\"chore: clean repository, move runtime files to runtime/ and update .gitignore\""
echo "Run:"
echo "    git add -A"
echo "    git commit -m \"chore: clean repository, move runtime files to runtime/ and update .gitignore\""
echo "    git push origin main"

echo "[Hygiene] Done."
