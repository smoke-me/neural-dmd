"""Regenerate summary.html for an existing experiment, without re-running
any analyse / plot ops.

Reads the experiment's metrics.json + manifest.json + any PNGs already
on disk, and rewrites summary.html through neural_dmd.experiments.write_summary().

Usage:

    python scripts/write_summary.py
        # rebuilds summary.html for the lexicographically-latest
        # experiment under outputs/experiments/

    NEURAL_DMD_EXP_ID=2026-05-08_19-52-31_baseline python scripts/write_summary.py
        # targets a specific experiment by id

    python scripts/write_summary.py --exp 2026-05-08_19-52-31_baseline
        # same, but via flag

Useful when you've edited write_summary() (layout/copy/CSS) and want to
re-render every existing experiment's report without paying the analyse
+ plot cost again.
"""

import argparse
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_dmd import experiments as E  # noqa: E402
from neural_dmd.log import log  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", type=str, default=None,
                        help="experiment id to rebuild (defaults to "
                             "NEURAL_DMD_EXP_ID env var, or the latest "
                             "existing experiment)")
    parser.add_argument("--all", action="store_true",
                        help="rebuild summary.html for EVERY experiment "
                             "under outputs/experiments/")
    args = parser.parse_args()

    if args.all:
        from neural_dmd import config as C
        exp_root = C.OUTPUT_ROOT / "experiments"
        if not exp_root.exists():
            log("warn", f"no experiments dir at {exp_root}")
            return
        eids = sorted(d.name for d in exp_root.iterdir() if d.is_dir())
        if not eids:
            log("warn", f"no experiments under {exp_root}")
            return
        for eid in eids:
            os.environ["NEURAL_DMD_EXP_ID"] = eid
            log("info", f"rebuilding summary for {eid}")
            E.write_summary()
        return

    if args.exp:
        os.environ["NEURAL_DMD_EXP_ID"] = args.exp
    # E.exp_id() resolves: env var > latest existing > fresh mint. The
    # "fresh mint" branch would create a new (empty) dir, which is the
    # opposite of what we want here; guard against it.
    eid = E.exp_id()
    if not E.metrics_path().exists():
        log("warn",
            f"no metrics.json at {E.metrics_path()} - did you mean to "
            f"point at a different exp id? (--exp <id>)")
        return
    log("info", f"rebuilding summary for {eid}")
    E.write_summary()


if __name__ == "__main__":
    main()
