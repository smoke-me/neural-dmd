"""
Experiment orchestrator.

  1. Mints a fresh exp_id (`<YYYY-MM-DD_HH-MM-SS>_<EXP_LABEL>`) and
     creates the experiment directory layout.
  2. Tees stdout + stderr into `outputs/experiments/<exp_id>/run.log`
     (ANSI colour codes are stripped from the file copy).
  3. Computes the training-data hash. Cache miss → invoke train.py.
     Cache hit → log "data cache HIT, reusing".
  4. For each method listed in `config.METHODS` (or `--methods`), runs
     the operations in `--only` (default: analyze + all 3 plots).
  5. Aggregates per-method metrics into `summary.md`.

# Most useful invocations

  python scripts/run_exp.py
        # full pipeline, all methods in config.METHODS, all 3 plots

  python scripts/run_exp.py --skip-loss
        # everything EXCEPT the slow loss plot (forward passes per
        # snapshot). Equivalent to: --only analyze,plot_eigenvalues,plot_accuracy

  python scripts/run_exp.py --methods dmdc,sdmdc
        # subset of methods, full pipeline per method

  python scripts/run_exp.py --only analyze,plot_loss
        # analyse + loss plot only; skip eigenvalues + accuracy

  python scripts/run_exp.py --methods dmdc --only plot_eigenvalues
        # one method, one plot (analysis must already exist in the
        # active experiment)

  python scripts/run_exp.py --force-train
        # retrain even on data cache hit

# Single-script invocations (live in scripts/<op>_<method>.py)

  python scripts/analyze_dmdc.py
  python scripts/plot_loss_dmdc.py
  python scripts/plot_accuracy_optdmdc.py
  python scripts/plot_eigenvalues_coptdmdc.py
        # each operates on the latest existing experiment, or use
        # NEURAL_DMD_EXP_ID=... env var to target a specific one.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

# Force utf-8 on terminal streams as early as possible. Windows defaults
# stdout/stderr to cp1252, which crashes on Greek letters (ρ, λ) etc.
# that show up in our logs and summary tables. Idempotent + safe to call
# again later inside tee_run_log.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_dmd import config as C
from neural_dmd import experiments as E
from neural_dmd.log import banner, log
from neural_dmd.runners import run_full_pipeline, _OPS_ORDER


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods",     type=str, default=None,
                        help="comma-separated method override (else config.METHODS)")
    parser.add_argument("--only",        type=str, default=None,
                        help=("comma-separated operations to run from "
                              + ",".join(_OPS_ORDER) + " (default: all)"))
    parser.add_argument("--skip-loss",   action="store_true",
                        help="shortcut for --only analyze,plot_eigenvalues,plot_accuracy "
                             "(skips both plot_loss and plot_classification - they share "
                             "the per-checkpoint forward-pass cost)")
    parser.add_argument("--force-train", action="store_true",
                        help="retrain even if data cache hit")
    args = parser.parse_args()

    methods = (tuple(m.strip() for m in args.methods.split(","))
               if args.methods else tuple(C.METHODS))

    if args.only:
        ops = tuple(o.strip() for o in args.only.split(","))
    elif args.skip_loss:
        ops = ("analyze", "plot_eigenvalues", "plot_accuracy")
    else:
        ops = None     # all

    # 1. mint experiment + tee stdout
    eid = E.init_exp()
    E.tee_run_log()
    banner("run_exp.py start",
           exp_id=eid, exp_label=C.EXP_LABEL,
           methods=",".join(methods),
           ops=",".join(ops) if ops else "all",
           data_hash=E.data_hash())
    if C.EXP_DESCRIPTION:
        log("info", f"description: {C.EXP_DESCRIPTION}")
    log("info", f"experiment dir: {E.exp_dir()}")

    t_total = time.time()

    # 2. data cache or train
    if E.data_cached() and not args.force_train:
        log("ok",
            f"data cache HIT for hash={E.data_hash()}  "
            f"(reusing {E.data_dir()})")
    else:
        if E.data_cached() and args.force_train:
            log("warn", "data cache hit but --force-train given; retraining")
        else:
            log("info", f"data cache MISS for hash={E.data_hash()}; training")
        spec = importlib.util.spec_from_file_location(
            "_train_mod",
            Path(__file__).parent / "train.py")
        mod = importlib.util.module_from_spec(spec)
        sys_argv_save = sys.argv
        sys.argv = ["train.py"] + (["--force"] if args.force_train else [])
        try:
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            mod.main()
        finally:
            sys.argv = sys_argv_save

    # 3. analyze + plots per method
    run_full_pipeline(methods=methods, ops=ops)

    # 4. summary
    E.write_summary()

    banner("run_exp.py done",
           exp_id=eid, total_seconds=f"{time.time() - t_total:.1f}",
           summary=E.summary_path())


if __name__ == "__main__":
    main()
