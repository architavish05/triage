"""Executable CLI runner for Phase 6 flowpath tracing."""

import sys
from src.flowpath.trace import main

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mock_run"
    main(target)
