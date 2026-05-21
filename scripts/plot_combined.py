"""Thin wrapper: run `runners.do_plot_combined(config.METHODS)`.

Reads the per-method eval-curve caches under
<exp_dir>/curves/<method>.npz and writes the cross-method overlay to
<exp_dir>/plots/overlay/combined.png. Requires plot_loss or
plot_classification to have run for every method first."""

import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_dmd import config as C
from neural_dmd.runners import do_plot_combined


if __name__ == "__main__":
    do_plot_combined(list(C.METHODS))
