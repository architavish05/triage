"""Executable CLI runner for Phase 7 rescue intelligence."""

import sys
from src.rescue.trapped_risk import main

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mock_run"
    main(target)
