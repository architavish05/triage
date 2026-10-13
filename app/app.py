"""Triage | Flood damage from satellite data.

Redirect entrypoint to main.
"""
import sys
from pathlib import Path

# Ensure both repo root and app dir are in sys.path
app_dir = Path(__file__).resolve().parent
repo_root = app_dir.parent
for p in (str(repo_root), str(app_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from main import main
except (ImportError, ModuleNotFoundError):
    from app.main import main

if __name__ == "__main__":
    main()
