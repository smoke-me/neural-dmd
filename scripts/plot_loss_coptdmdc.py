"""Thin wrapper: run `runners.do_plot_loss('coptdmdc')`. See neural_dmd/runners.py for logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_dmd.runners import do_plot_loss


if __name__ == "__main__":
    do_plot_loss("coptdmdc")
