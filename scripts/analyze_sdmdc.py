"""Thin wrapper: run `runners.do_analyze('sdmdc')`. See neural_dmd/runners.py for logic."""

import sys
from pathlib import Path

# Force utf-8 (Windows cp1252 crashes on ρ/λ/Δ etc).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_dmd.runners import do_analyze


if __name__ == "__main__":
    do_analyze("sdmdc")
