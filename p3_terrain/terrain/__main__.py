"""Executable CLI runner for Phase 3 terrain filtering."""

import sys

try:
    from p3_terrain.terrain.hand_slope import main
except ImportError:
    from src.terrain.hand_slope import main

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mock_run"
    main(target)
