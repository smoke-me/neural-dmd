"""Thin wrapper: run `runners.do_plot_classification('coptdmdc')`.
See neural_dmd/runners.py for logic."""

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_dmd.runners import do_plot_classification


if __name__ == "__main__":
    do_plot_classification("coptdmdc")
