"""Triage | Flood damage from satellite data.

Decision-support application for disaster response and evacuation prioritization.
Launch entrypoint: streamlit run app/main.py
"""

import sys
import os
from pathlib import Path

# Prepend project root and app directory to sys.path so package and components imports succeed
app_dir = Path(__file__).resolve().parent
repo_root = app_dir.parent
for p in (str(repo_root), str(app_dir), str(repo_root / "p4_app"), str(repo_root / "p4_app" / "app")):
    if p not in sys.path:
        sys.path.insert(0, p)

from p4_app.app.main import main

if __name__ == "__main__":
    main()
