"""
Experiment management: content-addressed training-data cache + per-run
experiment directories with manifests, configs, plots, metrics, and a
human-readable summary.

# Layout

    outputs/
      data/<data_hash>/                   training data, content-addressed
        snapshots.npz
        model.pt
        train_config.json
        train.log
      experiments/<exp_id>/                one per analysis run
        manifest.json                      exp_id, label, description, time, data_hash, methods
        config.json                        full config snapshot at run-time
        run.log                            tee'd stdout/stderr of the run
        metrics.json                       per-method metrics, written incrementally
        analyses/<method>.npz              one per method actually run
        plots/<method>/{loss,accuracy,eigenvalues}.png
        summary.html                       self-contained HTML report (PNGs
                                           inlined as base64 data URLs)

# Identification

    data_hash  -- sha256(training-relevant config) truncated to 12 hex.
                  TRAIN_FIELDS lists exactly which config attributes are
                  hashed; changing any of them invalidates the cache and
                  triggers a fresh training run.
    exp_id     -- "<YYYY-MM-DD_HH-MM-SS>_<EXP_LABEL>". Always sortable;
                  always unique within a wall-clock second.

# Active experiment

The "active" experiment id is held in the NEURAL_DMD_EXP_ID env var.
- run_exp.py calls init_exp() at startup, which mints a fresh id and
  publishes it via this env var so child scripts inherit it.
- Standalone scripts (analyze_*.py, plot_*.py) call get_or_init_exp()
  which uses the env var if present, else mints a new id.

# Public surface

    data_hash() -> str
    data_dir() / snapshots_path() / model_path() / data_log_path()
    data_cached() -> bool
    train_config_dict() -> dict   exact subset that goes into the hash

    init_exp() -> str             mint new exp_id + dirs + manifest
    get_or_init_exp() -> str      use env var or init_exp()
    exp_id() / exp_dir()
    analysis_path(method) / plot_path(method, kind) / plot_dir(method)
    run_log_path() / summary_path() / manifest_path() / config_snapshot_path()
    metrics_path()

    write_metric(method, key, value)   incremental metric storage
    read_metrics() -> dict
    write_summary()                     aggregate metrics into summary.html
                                        (self-contained, base64-embedded plots)

    log_eigenvalue_report(label, eigvals, *, tol=None, top_n=10) -> dict
        Verbose eigenvalue diagnostic + returns metrics for summary.

    snapshot_full_config() -> dict     all-uppercase attrs of config module
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import config as C
from .log import log


# ---------------------------------------------------------------------------
# fields whose change invalidates the training-data cache
#
# Anything that affects the snapshots.npz / model.pt content goes here.
# Anything else (RANK, METHOD_PARAMS, EXP_LABEL, ...) does NOT - those
# are analysis-time choices and should reuse cached data.
# ---------------------------------------------------------------------------

TRAIN_FIELDS = (
    "DATASET",
    "ARCH",
    "EPOCHS",
    "EXTRA_STEPS",
    "BATCH_SIZE",
    "LR",
    "LR_MIN",
    "LOSS",
    "OPTIMIZER",
    "OPTIMIZER_PARAMS",
    "CONTROLS",
    "CONTROL_PARAMS",
    "FIT_FRAC",
    "FIT_RANGE",
    "SNAP_FIT_EVERY",
    "SNAP_FORECAST_EVERY",
    "SEED",
    "NOISE_SIGMA",
)

_EXP_ID_ENV = "NEURAL_DMD_EXP_ID"


# ---------------------------------------------------------------------------
# data hash + paths
# ---------------------------------------------------------------------------

def resolve_fit_window(total_steps: int) -> tuple[int, int]:
    """Return (fit_start_step, fit_end_step) from config.

    Exactly ONE of <code>FIT_FRAC</code> / <code>FIT_RANGE</code> must
    be set (the other must be <code>None</code>). Setting both, or
    setting neither, is a configuration error and raises ValueError -
    the previous "FIT_RANGE wins silently" behaviour was easy to misread.

    Accepted forms:

      FIT_FRAC = f               float in (0, 1]
                                 -> [0, round(f * total_steps))
      FIT_FRAC = (a, b)          floats in [0, 1], a < b
                                 -> [round(a*N), round(b*N))
      FIT_RANGE = (start, end)   ints in [0, total_steps], start < end
                                 -> [start, end)

    Both windows must contain >= 2 snapshots (the DMDc fit needs at
    least one pair); we round end up to satisfy that.
    """
    fr = getattr(C, "FIT_RANGE", None)
    ff = getattr(C, "FIT_FRAC",  None)

    if fr is not None and ff is not None:
        raise ValueError(
            "config error: exactly one of FIT_FRAC / FIT_RANGE may be set "
            f"(both are: FIT_FRAC={ff!r}, FIT_RANGE={fr!r}). Set the "
            "inactive one to None.")
    if fr is None and ff is None:
        raise ValueError(
            "config error: one of FIT_FRAC or FIT_RANGE must be set; both "
            "are None. Use FIT_FRAC (fraction-based) or FIT_RANGE "
            "(step-based) to define the fit window.")

    if fr is not None:
        if not (isinstance(fr, (tuple, list)) and len(fr) == 2):
            raise ValueError(
                f"FIT_RANGE must be a (start, end) pair; got {fr!r}")
        start, end = int(fr[0]), int(fr[1])
        if not (0 <= start < end <= total_steps):
            raise ValueError(
                f"FIT_RANGE={fr} invalid for total_steps={total_steps}; "
                f"need 0 <= start < end <= {total_steps}")
        return start, end

    # FIT_FRAC branch
    if isinstance(ff, (tuple, list)):
        if len(ff) != 2:
            raise ValueError(
                f"FIT_FRAC tuple must be (start_frac, end_frac); got {ff!r}")
        a, b = float(ff[0]), float(ff[1])
    else:
        a, b = 0.0, float(ff)
    if not (0.0 <= a < b <= 1.0):
        raise ValueError(
            f"FIT_FRAC={ff!r} invalid; need 0 <= start_frac < end_frac <= 1 "
            "(or a single float in (0, 1]).")
    start = int(round(a * total_steps))
    end   = int(round(b * total_steps))
    end   = max(end, start + 2)
    end   = min(end, total_steps)
    if not (start < end):
        raise ValueError(
            f"FIT_FRAC={ff!r} on total_steps={total_steps} yields empty "
            f"window [{start}, {end})")
    return start, end


def train_config_dict() -> dict:
    """Subset of config used to derive the data_hash.

    Special handling for the fit-window selector:
        Only the ACTIVE knob (FIT_RANGE or FIT_FRAC, whichever is not
        None) participates in the hash. Toggling the inactive one or
        setting it to None doesn't invalidate the cache.

        Mutual exclusivity is enforced lazily here too: if both are set
        or both are None we surface the same ValueError as the runtime
        resolver, so the failure happens before we mint a phantom hash.
    """
    fields: dict[str, Any] = {}
    for k in TRAIN_FIELDS:
        if k in ("FIT_FRAC", "FIT_RANGE",
                 "OPTIMIZER_PARAMS", "CONTROL_PARAMS"):
            continue   # handled below
        v = getattr(C, k, None)
        if isinstance(v, list):
            v = tuple(v)
        fields[k] = v

    # OPTIMIZER_PARAMS: only the entry for the active OPTIMIZER goes
    # into the hash, so toggling kwargs for an unused optimizer does
    # NOT invalidate the cache. Mirrors the FIT_FRAC/FIT_RANGE pattern.
    opt_name = getattr(C, "OPTIMIZER", "adam")
    opt_all  = getattr(C, "OPTIMIZER_PARAMS", {}) or {}
    opt_kw   = opt_all.get(opt_name, {}) or {}
    fields["OPTIMIZER_PARAMS"] = {
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in sorted(opt_kw.items())
    }

    # CONTROL_PARAMS: same pattern - only kwargs for sources actually
    # listed in CONTROLS contribute. Editing an inactive entry doesn't
    # change the snapshot data.
    ctrl_names = tuple(getattr(C, "CONTROLS", ("lr",)))
    ctrl_all   = getattr(C, "CONTROL_PARAMS", {}) or {}
    fields["CONTROLS"] = ctrl_names
    fields["CONTROL_PARAMS"] = {
        name: {
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in sorted((ctrl_all.get(name, {}) or {}).items())
        }
        for name in ctrl_names
    }

    fr = getattr(C, "FIT_RANGE", None)
    ff = getattr(C, "FIT_FRAC",  None)
    if fr is not None and ff is not None:
        raise ValueError(
            "config error: exactly one of FIT_FRAC / FIT_RANGE may be set "
            f"(both are: FIT_FRAC={ff!r}, FIT_RANGE={fr!r}). Set the "
            "inactive one to None.")
    if fr is None and ff is None:
        raise ValueError(
            "config error: one of FIT_FRAC or FIT_RANGE must be set; both "
            "are None.")
    if fr is not None:
        fields["FIT_RANGE"] = tuple(fr) if isinstance(fr, (list, tuple)) else fr
    else:
        fields["FIT_FRAC"] = (tuple(float(x) for x in ff)
                              if isinstance(ff, (list, tuple))
                              else float(ff))

    return fields


def data_hash() -> str:
    """sha256 of training-relevant config, truncated to 12 hex chars."""
    blob = json.dumps(train_config_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def data_dir() -> Path:
    d = C.OUTPUT_ROOT / "data" / data_hash()
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshots_path() -> Path:
    return data_dir() / "snapshots.npz"


def model_path() -> Path:
    return data_dir() / "model.pt"


def train_config_path() -> Path:
    return data_dir() / "train_config.json"


def data_log_path() -> Path:
    return data_dir() / "train.log"


def data_cached() -> bool:
    """True iff the training-data cache hit covers everything analyze
    needs (snapshots + model)."""
    return snapshots_path().exists() and model_path().exists()


# ---------------------------------------------------------------------------
# experiment id + dirs
# ---------------------------------------------------------------------------

def _make_exp_id() -> str:
    label = getattr(C, "EXP_LABEL", "default") or "default"
    # sanitise: keep alnum, dash, underscore, dot
    safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in label)
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{ts}_{safe}"


def exp_id() -> str:
    """Currently-active experiment id, in order of preference:

      1. NEURAL_DMD_EXP_ID env var (set by run_exp.py for child scripts,
         or by the user to target a specific historical experiment).
      2. The lexicographically-latest existing experiment under
         outputs/experiments/. exp_ids start with `<YYYY-MM-DD_HH-MM-SS>`
         so lexicographic order = chronological order. Useful for ad-hoc
         re-runs of a single plot script after run_exp.py has produced
         the analyses.
      3. A freshly-minted exp_id (creates a new directory).
    """
    if _EXP_ID_ENV in os.environ:
        return os.environ[_EXP_ID_ENV]
    exp_root = C.OUTPUT_ROOT / "experiments"
    if exp_root.exists():
        existing = sorted(d.name for d in exp_root.iterdir() if d.is_dir())
        if existing:
            eid = existing[-1]
            os.environ[_EXP_ID_ENV] = eid
            return eid
    eid = _make_exp_id()
    os.environ[_EXP_ID_ENV] = eid
    return eid


def exp_dir() -> Path:
    d = C.OUTPUT_ROOT / "experiments" / exp_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


def analysis_dir() -> Path:
    d = exp_dir() / "analyses"
    d.mkdir(exist_ok=True)
    return d


def analysis_path(method: str) -> Path:
    return analysis_dir() / f"{method}.npz"


def plot_dir(method: str) -> Path:
    d = exp_dir() / "plots" / method
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_path(method: str, kind: str) -> Path:
    return plot_dir(method) / f"{kind}.png"


def curves_dir() -> Path:
    """Per-method eval-curve cache (loss + classification metrics over
    the eval grid for each method's predicted network). See
    runners._pred_eval_curve_cached for the schema."""
    d = exp_dir() / "curves"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_log_path() -> Path:
    return exp_dir() / "run.log"


def summary_path() -> Path:
    return exp_dir() / "summary.html"


def manifest_path() -> Path:
    return exp_dir() / "manifest.json"


def config_snapshot_path() -> Path:
    return exp_dir() / "config.json"


def metrics_path() -> Path:
    return exp_dir() / "metrics.json"


# ---------------------------------------------------------------------------
# experiment lifecycle
# ---------------------------------------------------------------------------

def snapshot_full_config() -> dict:
    """Full config module snapshot - every UPPERCASE attribute."""
    out: dict[str, Any] = {}
    for name in dir(C):
        if name.startswith("_") or not name[0].isupper():
            continue
        try:
            val = getattr(C, name)
        except AttributeError:
            continue
        # drop things that aren't json-serialisable cleanly
        try:
            json.dumps(val, default=str)
            out[name] = val
        except Exception:
            out[name] = str(val)
    return out


def init_exp() -> str:
    """Mint a fresh exp_id, create directories, and write manifest +
    config snapshot. Sets NEURAL_DMD_EXP_ID env var so child processes
    (analyze, plot) inherit the active experiment.

    Returns the exp_id."""
    eid = _make_exp_id()
    os.environ[_EXP_ID_ENV] = eid
    d = exp_dir()
    methods = list(getattr(C, "METHODS", ("dmdc",)))
    manifest = {
        "exp_id":         eid,
        "label":          getattr(C, "EXP_LABEL", "default"),
        "description":    getattr(C, "EXP_DESCRIPTION", ""),
        "iso_time":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "unix_time":      time.time(),
        "data_hash":      data_hash(),
        "methods":        methods,
    }
    manifest_path().write_text(json.dumps(manifest, indent=2),
                               encoding="utf-8")
    config_snapshot_path().write_text(
        json.dumps(snapshot_full_config(), indent=2, default=str),
        encoding="utf-8")
    log("info", f"experiments: init exp_id={eid}  data_hash={data_hash()}  dir={d}")
    return eid


def get_or_init_exp() -> str:
    """Active exp_id resolution for standalone scripts (analyze_*.py,
    plot_*.py invoked directly):

      1. NEURAL_DMD_EXP_ID env var if set.
      2. Latest-existing experiment under outputs/experiments/ (so a
         re-run of a single plot script after run_exp.py picks up the
         fresh experiment automatically).
      3. Fresh init_exp() (writes manifest + config snapshot).
    """
    if _EXP_ID_ENV in os.environ:
        return os.environ[_EXP_ID_ENV]
    exp_root = C.OUTPUT_ROOT / "experiments"
    if exp_root.exists():
        existing = sorted(d.name for d in exp_root.iterdir() if d.is_dir())
        if existing:
            eid = existing[-1]
            os.environ[_EXP_ID_ENV] = eid
            log("info", f"experiments: reusing latest existing exp_id={eid}")
            return eid
    return init_exp()


# ---------------------------------------------------------------------------
# tee logging into the active experiment dir
# ---------------------------------------------------------------------------

import re

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHFA-Z]")


class _Tee:
    """Write to two streams; optionally strip ANSI escape codes for
    streams that go to disk (so run.log is readable in plain editors).

    Constructor takes a list of (stream, strip_ansi) tuples.
    """

    def __init__(self, *streams_and_modes):
        # accepts either (stream,) or (stream, strip_ansi_bool); the
        # bare-stream form preserves color (default for terminals).
        self._items = []
        for item in streams_and_modes:
            if isinstance(item, tuple):
                self._items.append(item)
            else:
                self._items.append((item, False))

    def write(self, s):
        s_clean: str | None = None
        for st, strip in self._items:
            try:
                if strip:
                    if s_clean is None:
                        s_clean = _ANSI_RE.sub("", s)
                    st.write(s_clean)
                else:
                    st.write(s)
            except Exception:
                pass

    def flush(self):
        for st, _ in self._items:
            try:
                st.flush()
            except Exception:
                pass


def tee_run_log() -> None:
    """Mirror sys.stdout / sys.stderr into the experiment's run.log.
    The terminal stream keeps ANSI colour; the run.log stream has the
    escape codes stripped so the log file is readable in plain text
    editors. Idempotent.

    Forces utf-8 on every writer so non-ASCII (ρ, λ, Δ, ←, →, etc.)
    survives:
      - the on-disk run.log is opened with encoding='utf-8'
      - the original sys.stdout/sys.stderr (terminal) are reconfigured
        to utf-8 with errors='replace', avoiding Windows cp1252 crashes
    """
    if getattr(sys.stdout, "_neural_dmd_tee", False):
        return

    # Reconfigure terminal streams to utf-8 so prints of ρ / λ / Δ etc.
    # don't crash on Windows (default cp1252) or any other 8-bit locale.
    # `errors="replace"` keeps the program alive even on truly exotic
    # output streams that still can't represent a particular char.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    fp = open(run_log_path(), "a", buffering=1,
              encoding="utf-8", errors="replace")     # line-buffered, utf-8
    fp.write(f"\n# === run.log opened at {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
    new_out = _Tee((sys.stdout, False), (fp, True))
    new_out._neural_dmd_tee = True                     # type: ignore[attr-defined]
    sys.stdout = new_out                               # type: ignore[assignment]
    sys.stderr = _Tee((sys.stderr, False), (fp, True)) # type: ignore[assignment]


# ---------------------------------------------------------------------------
# metrics + summary
# ---------------------------------------------------------------------------

def write_metric(method: str, key: str, value) -> None:
    """Append (method, key) -> value to the experiment's metrics.json.
    Value is JSON-serialisable (str fallback for anything weird)."""
    p = metrics_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8", errors="replace")) if p.exists() else {}
    except Exception:
        d = {}
    d.setdefault(method, {})[key] = value
    p.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def write_metrics_bulk(method: str, mapping: dict) -> None:
    p = metrics_path()
    try:
        d = json.loads(p.read_text(encoding="utf-8", errors="replace")) if p.exists() else {}
    except Exception:
        d = {}
    d.setdefault(method, {}).update(mapping)
    p.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")


def read_metrics() -> dict:
    p = metrics_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# eigenvalue diagnostic
# ---------------------------------------------------------------------------

def log_eigenvalue_report(label: str, eigvals, *,
                          tol: float | None = None,
                          top_n: int = 10) -> dict:
    """Verbose eigenvalue analysis for a discrete-time operator A:
        - count inside / outside the unit circle (with tolerance)
        - spectral radius
        - list every eigenvalue outside the circle (full precision)
        - list the top-|λ| values (whether stable or not)
    Returns a metrics dict suitable for write_metrics_bulk()."""
    if tol is None:
        tol = getattr(C, "EIG_STABLE_TOL", 1e-12)
    eigvals = np.asarray(eigvals).astype(np.complex128)
    mags = np.abs(eigvals)
    n = len(mags)
    in_mask  = mags <= 1.0 + tol
    out_mask = ~in_mask
    n_in     = int(in_mask.sum())
    n_out    = int(out_mask.sum())
    sr       = float(mags.max()) if n else 0.0

    log("info",
        f"{label}: {n} eigenvalues  inside={n_in}  outside={n_out}  "
        f"(tol={tol:.0e})  spectral_radius={sr:.6f}")

    outside_records = []
    if n_out > 0:
        log("warn", f"{label}: {n_out} eigenvalue(s) OUTSIDE unit circle:")
        out_idx = np.where(out_mask)[0]
        out_idx = out_idx[np.argsort(-mags[out_idx])]
        for i in out_idx[:top_n]:
            log("warn",
                f"  λ_{int(i):4d} = {eigvals[i].real:+.6f}{eigvals[i].imag:+.6f}j   "
                f"|λ|={mags[i]:.6f}  excess={(mags[i] - 1.0):+.3e}")
            outside_records.append({
                "index":     int(i),
                "real":      float(eigvals[i].real),
                "imag":      float(eigvals[i].imag),
                "magnitude": float(mags[i]),
                "excess":    float(mags[i] - 1.0),
            })
        if len(out_idx) > top_n:
            log("warn", f"  (+{len(out_idx) - top_n} more outside; truncated to top {top_n})")

    if n > 0:
        order = np.argsort(-mags)[:top_n]
        log("info", f"{label}: top {min(top_n, n)} by |λ|:")
        for j in order:
            marker = "!" if mags[j] > 1.0 + tol else " "
            log("info",
                f"  {marker} λ_{int(j):4d} = "
                f"{eigvals[j].real:+.6f}{eigvals[j].imag:+.6f}j   "
                f"|λ|={mags[j]:.6f}")

    return {
        "n_total":           n,
        "n_inside":          n_in,
        "n_outside":         n_out,
        "spectral_radius":   sr,
        "stable_tol":        tol,
        "outside_top":       outside_records,
    }


# ---------------------------------------------------------------------------
# summary.md
# ---------------------------------------------------------------------------

# Canonical ordering of the test-set metrics shown in the summary
# tables. (key, display name). Mirrors runners._CURVE_METRICS.
_TEST_METRICS = [
    ("loss",      "loss"),
    ("accuracy",  "accuracy"),
    ("precision", "precision"),
    ("recall",    "recall"),
    ("f1",        "F1"),
]

# Per-method stability + cost columns shown alongside the forecasted
# test metrics. Pure-DMDc methods (no LM) leave the LM columns blank.
_STABILITY_KEYS = [
    ("rank",            "rank"),
    ("spectral_radius", "spectral ρ"),
    ("n_outside",       "λ>1"),
    ("lm_iters",        "LM iters"),
    ("lm_stop_reason",  "LM stop"),
    ("wall_seconds",    "wall (s)"),
]


def _fmt_cell(key: str, val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        if math.isinf(val):
            return "∞" if val > 0 else "−∞"
        # very small or very large -> scientific notation for readability
        if val != 0.0 and (abs(val) < 1e-3 or abs(val) >= 1e6):
            return f"{val:.3e}"
        if "pct" in key:
            return f"{val:.2f}"
        if "wall" in key:
            return f"{val:.1f}"
        # signed display for the forecast-vs-real gap columns
        if key.startswith("gap_"):
            return f"{val:+.4f}"
        # all [0,1] test metrics + loss + spectral radius land here
        if any(tag in key
               for tag in ("loss", "accuracy", "precision", "recall",
                           "f1", "spectral")):
            return f"{val:.4f}"
        return f"{val:.4g}"
    return str(val)


def _classify_cell(key: str, val) -> str:
    """Inline CSS class for table cells based on key + value semantics."""
    if val is None:
        return "num"
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return "num bad"
    if key == "n_outside":
        return "num bad" if (isinstance(val, (int, float)) and val > 0) else "num good"
    return "num"


_REPORT_CSS = """
:root {
  --fg: #1f2937;
  --fg-muted: #4b5563;
  --bg: #f8fafc;
  --bg-card: #ffffff;
  --border: #d1d5db;
  --border-soft: #e5e7eb;
  --accent: #2563eb;
  --accent-soft: #dbeafe;
  --good: #15803d;
  --bad: #b91c1c;
  --code-bg: #f1f5f9;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  font-size: 15px;
  color: var(--fg);
  background: var(--bg);
  line-height: 1.55;
}
main {
  max-width: 1200px;
  margin: 2em auto;
  padding: 0 1.5em 4em;
}
h1, h2, h3, h4 {
  color: #0f172a;
  margin: 1.4em 0 0.5em;
}
h1 {
  font-size: 1.75em;
  border-bottom: 3px solid var(--accent);
  padding-bottom: 0.3em;
  margin-top: 0;
}
h2 {
  font-size: 1.3em;
  border-bottom: 1px solid var(--border-soft);
  padding-bottom: 0.25em;
  margin-top: 2.2em;
}
h3 {
  font-size: 1.1em;
  margin-top: 1.5em;
}
code {
  background: var(--code-bg);
  padding: 0.12em 0.4em;
  border-radius: 3px;
  font-size: 0.92em;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.description {
  background: var(--bg-card);
  border-left: 3px solid var(--accent);
  padding: 0.9em 1.2em;
  margin: 1em 0 1.5em;
  font-style: italic;
  color: var(--fg-muted);
  border-radius: 4px;
}
dl.meta {
  display: grid;
  grid-template-columns: max-content 1fr;
  column-gap: 1.2em;
  row-gap: 0.4em;
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 1em 1.2em;
  margin: 1em 0 1.5em;
}
dl.meta dt {
  font-weight: 600;
  color: var(--fg-muted);
}
dl.meta dd { margin: 0; }
table {
  border-collapse: collapse;
  margin: 1em 0;
  width: 100%;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: hidden;
}
th, td {
  padding: 0.5em 0.8em;
  text-align: left;
  border-bottom: 1px solid var(--border-soft);
  border-right: 1px solid var(--border-soft);
}
th:last-child, td:last-child { border-right: none; }
tr:last-child td { border-bottom: none; }
th {
  background: var(--accent-soft);
  color: #1e3a8a;
  font-weight: 600;
  font-size: 0.92em;
}
td.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.95em;
}
td.bad { color: var(--bad); font-weight: 600; }
td.good { color: var(--good); font-weight: 600; }
td.method-cell { font-weight: 600; }
tr.best-rank td {
  background: #ecfdf5;
  font-weight: 600;
  color: var(--good);
}
ul.eigvals {
  list-style: none;
  margin: 0.5em 0 1em;
  padding: 0.7em 1em;
  background: var(--code-bg);
  border-radius: 4px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.9em;
}
ul.eigvals li { padding: 0.15em 0; }
.stable-note {
  margin: 0.4em 0;
  padding: 0.4em 0.8em;
  background: #ecfdf5;
  border-left: 3px solid var(--good);
  border-radius: 2px;
  font-size: 0.95em;
}
.lm-note {
  margin: 0.3em 0;
  padding: 0.4em 0.8em;
  background: var(--bg-card);
  border-left: 3px solid var(--accent);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.9em;
}
section.method-plots {
  margin: 1.2em 0;
  padding: 1em 1.2em;
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
}
section.method-plots h3 { margin-top: 0; }
.plot { margin: 1em 0; }
.plot img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--border-soft);
  border-radius: 4px;
}
.plot-caption {
  font-size: 0.88em;
  color: var(--fg-muted);
  margin-top: 0.3em;
  text-align: center;
}
.note {
  font-size: 0.92em;
  color: var(--fg-muted);
  margin: 0.4em 0 1em;
}
dl.glossary {
  display: grid;
  grid-template-columns: max-content 1fr;
  column-gap: 1.2em;
  row-gap: 0.55em;
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 1em 1.2em;
  margin: 1em 0 1.5em;
  font-size: 0.95em;
}
dl.glossary dt {
  font-weight: 600;
  color: var(--accent);
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
}
dl.glossary dd {
  margin: 0;
  color: var(--fg);
}
details.config-dump {
  background: var(--bg-card);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 0.6em 1em;
  margin: 1em 0 1.5em;
}
details.config-dump summary {
  cursor: pointer;
  padding: 0.2em 0;
  outline: none;
}
details.config-dump table { margin-top: 0.8em; }
details.config-dump pre {
  background: var(--code-bg);
  margin: 0;
  padding: 0.5em 0.7em;
  border-radius: 4px;
  font-size: 0.86em;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
.cm-strip {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: 1em;
  margin: 1em 0 1.5em;
}
.cm-strip > div { min-width: 0; }
.cm-strip table { margin: 0.3em 0; font-size: 0.88em; }
.cm-strip th, .cm-strip td { padding: 0.3em 0.5em; }
.cm-strip h4 {
  margin: 0 0 0.3em;
  font-size: 1.0em;
}
""".strip()


# Per-method display label looked up from the methods registry; falls
# back to the raw key when the method isn't (or no longer) registered.
def _method_label(method: str) -> str:
    try:
        from .methods import get as _get
        return _get(method).get("label", method)
    except Exception:
        return method


# Captions for the per-method plot section. One-line layman gloss so
# the report reader knows what they're looking at without having to
# trace it back to the source code.
_PLOT_CAPTIONS = {
    "loss": ("Test-loss curves: the real network's test loss at each "
             "recorded training step (cyan), and the forecast's loss "
             "when its predicted weights are plugged into the same "
             "network (dashed orange). Lower is better. The dashed "
             "vertical line marks where the fit window ends and the "
             "forecast begins."),
    "classification": ("Four classification metrics evaluated on the "
                       "test set at every recorded step, for the real "
                       "network vs the forecasted network. All four "
                       "live in [0, 1]; higher is better. The dashed "
                       "vertical line marks the fit/forecast boundary."),
    "accuracy": ("How close the forecast's parameter vector is to the "
                 "real one in raw L2 distance, expressed as a percentage. "
                 "100% means an exact match. This is reconstruction "
                 "quality in weight space, NOT classification accuracy."),
    "eigenvalues": ("Spectrum of the linear operator the method "
                    "learns. Eigenvalues inside the dashed unit circle "
                    "decay (stable forecast); outside they grow "
                    "exponentially (forecast eventually diverges)."),
}


def _metric_glossary_html() -> str:
    """Concise, dataset-agnostic definitions for the test-set metrics
    used throughout the report. Rendered as a definition list inside
    the "What this report shows" section so the rest of the report can
    refer to these terms without re-defining them."""
    loss_name = str(getattr(C, "LOSS", "cross_entropy")).replace("_", " ")
    items = [
        ("loss",
         f"Here, <strong>{html.escape(loss_name)}</strong> on the test set. "
         "Measures how confident-yet-wrong the network's predicted "
         "probabilities are. 0 = perfect; grows without bound as the "
         "network becomes more confidently wrong. Lower is better."),
        ("accuracy",
         "Fraction of test items the network classifies correctly. "
         "In [0, 1]; higher is better. Easy to read, but blind to "
         "class imbalance."),
        ("precision (macro)",
         "Of all items the network labels as class <em>c</em>, what "
         "fraction actually are <em>c</em>. Low precision ⇒ many false "
         "alarms. We report the <em>macro</em> average: precision is "
         "computed per class then averaged with equal weight, so rare "
         "classes count as much as common ones. In [0, 1]; higher is better."),
        ("recall (macro)",
         "Of all items that actually are class <em>c</em>, what fraction "
         "the network catches. Low recall ⇒ many misses. Macro-averaged "
         "across classes. In [0, 1]; higher is better."),
        ("F1 (macro)",
         "Harmonic mean of precision and recall. Punishes models that "
         "score high on one but not the other. Macro-averaged across "
         "classes. In [0, 1]; higher is better."),
        ("confusion matrix",
         "Table where cell <code>(i, j)</code> counts the test items "
         "whose true class is <em>i</em> and that the network predicted "
         "as class <em>j</em>. The diagonal is correct predictions; "
         "everything off-diagonal is a mistake. The pattern of off-"
         "diagonal mass tells you <em>which</em> classes the network "
         "confuses, not just how often it errs."),
    ]
    parts = ['<dl class="glossary">']
    for term, body in items:
        parts.append(f"<dt>{html.escape(term)}</dt><dd>{body}</dd>")
    parts.append("</dl>")
    return "\n".join(parts)


def _is_confusion_matrix(cm) -> bool:
    """True iff `cm` looks like a non-empty square list-of-lists."""
    return (isinstance(cm, list) and len(cm) > 0
            and isinstance(cm[0], list) and len(cm[0]) == len(cm))


def _render_confusion_table(cm, h) -> str:
    """Render a square confusion-matrix list-of-lists as an HTML table.
    Diagonal cells get the 'good' colour class; everything else stays
    neutral so off-diagonal mass reads as the visual signal."""
    n = len(cm)
    parts = ["<table>"]
    parts.append("<thead><tr><th></th>")
    for j in range(n):
        parts.append(f"<th>p={h(str(j))}</th>")
    parts.append("</tr></thead><tbody>")
    for i, row in enumerate(cm):
        parts.append("<tr>")
        parts.append(f"<td class='method-cell'>t={h(str(i))}</td>")
        for j, v in enumerate(row):
            cls = "num good" if i == j and v > 0 else "num"
            parts.append(f"<td class='{cls}'>{h(_fmt_cell('cm', v))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _render_config_dump(h) -> str:
    """Read the on-disk config.json snapshot for this experiment (written
    by init_exp()) and emit it as a collapsible <details> block. Falls
    back to a live snapshot if the file is missing."""
    try:
        text = config_snapshot_path().read_text(encoding="utf-8",
                                                errors="replace")
        cfg  = json.loads(text)
    except Exception:
        try:
            cfg = snapshot_full_config()
        except Exception as e:
            return (f'<p class="note">(config snapshot unavailable: '
                    f"{h(str(e))})</p>")

    # Format each (key, value) row. Strings get a code-style cell; bigger
    # nested structures (lists, dicts, control_fn source) collapse into
    # a <pre> for readability while still being scannable in the doc.
    def _fmt_value(v) -> str:
        if isinstance(v, str) and "\n" in v:
            return f"<pre>{h(v)}</pre>"
        if isinstance(v, (list, tuple, dict)):
            return f"<pre>{h(json.dumps(v, indent=2, default=str))}</pre>"
        if isinstance(v, bool):
            return f"<code>{h(str(v))}</code>"
        if isinstance(v, (int, float)):
            return f"<code>{h(_fmt_cell('cfg', float(v)) if isinstance(v, float) else str(v))}</code>"
        if v is None:
            return "<code>None</code>"
        return f"<code>{h(str(v))}</code>"

    parts = []
    parts.append('<details class="config-dump"><summary>'
                 "<strong>Show full configuration snapshot</strong>  "
                 "<span class='note' style='font-weight:normal;'>"
                 "(every UPPERCASE attribute of <code>neural_dmd.config</code> "
                 "at the moment this experiment was launched; what you would "
                 "need to reproduce this run byte-for-byte)"
                 "</span></summary>")
    parts.append("<table>")
    parts.append("<thead><tr><th>Key</th><th>Value</th></tr></thead><tbody>")
    for key in sorted(cfg.keys()):
        parts.append(f"<tr><td class='method-cell'><code>{h(key)}</code></td>"
                     f"<td>{_fmt_value(cfg[key])}</td></tr>")
    parts.append("</tbody></table></details>")
    return "\n".join(parts)


def write_summary() -> None:
    """Aggregate metrics.json + manifest into a self-contained HTML
    report at <exp_dir>/summary.html. PNG plots are inlined as base64
    data URLs so the file renders correctly in any browser regardless
    of where it's moved.

    Layout (kept tight so values do not repeat across tables):

      1. Header + experiment configuration (run metadata)
      2. "What this report shows" - layman intro
      3. Real neural network - final test metrics + confusion matrix
      4. Forecasted neural networks - final test metrics by method
      5. Operator stability - rank, spectral radius, LM convergence
      6. Rank scan (if the auto scan ran)
      7. Plots - cross-method overlay then per-method PNG grid
    """
    metrics = read_metrics()
    if not metrics:
        log("warn", "experiments.write_summary: no metrics to write")
        return

    try:
        manifest = json.loads(
            manifest_path().read_text(encoding="utf-8", errors="replace"))
    except Exception:
        manifest = {}

    eid       = exp_id()
    exp_label = manifest.get("label", "default")
    desc      = manifest.get("description", "")
    iso_time  = manifest.get("iso_time", "")
    dh        = manifest.get("data_hash", data_hash())

    methods_cfg = list(getattr(C, "METHODS", tuple(metrics.keys())))
    # Method-row entries: ignore any "__meta__" keys (e.g. __rank_scan__,
    # __actual__), which are rendered in their own dedicated sections.
    metric_methods = [m for m in metrics if not m.startswith("__")]
    methods = [m for m in methods_cfg if m in metric_methods] + \
              [m for m in metric_methods if m not in methods_cfg]

    h = html.escape

    out: list[str] = []
    out.append("<!doctype html>")
    out.append('<html lang="en">')
    out.append("<head>")
    out.append('<meta charset="utf-8">')
    out.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    out.append(f"<title>Experiment: {h(eid)}</title>")
    out.append(f"<style>{_REPORT_CSS}</style>")
    out.append("</head>")
    out.append("<body><main>")

    # ----- 1. Header -----
    out.append(f"<h1>Experiment: <code>{h(eid)}</code></h1>")
    if desc:
        out.append(f'<div class="description">{h(desc)}</div>')

    noise = float(getattr(C, "NOISE_SIGMA", 0.0) or 0.0)
    fit_range = getattr(C, "FIT_RANGE", None)
    fit_frac  = getattr(C, "FIT_FRAC", None)
    # Source-of-truth description: which knob is active + the raw value.
    # The resolved (start_step, end_step) is recovered from the snapshot
    # cache below so the report shows both the knob the user set and
    # the actual window it produced.
    if fit_range is not None and fit_frac is None:
        fit_knob_str = f"FIT_RANGE = ({int(fit_range[0])}, {int(fit_range[1])})"
    elif fit_frac is not None and fit_range is None:
        if isinstance(fit_frac, (list, tuple)):
            a, b = float(fit_frac[0]), float(fit_frac[1])
            fit_knob_str = f"FIT_FRAC = ({a:.3f}, {b:.3f})"
        else:
            fit_knob_str = f"FIT_FRAC = {float(fit_frac):.3f}  (start = 0.0)"
    else:
        fit_knob_str = (f"⚠ both FIT_FRAC and FIT_RANGE set "
                        f"({fit_frac!r}, {fit_range!r})  "
                        if fit_range is not None and fit_frac is not None
                        else "⚠ neither FIT_FRAC nor FIT_RANGE set")

    # Resolved [start, end) in absolute step numbers, pulled from the
    # snapshot cache so the report shows what the recorder actually saw.
    try:
        snap_meta = np.load(snapshots_path()) if snapshots_path().exists() else None
        if snap_meta is not None:
            fs = int(snap_meta["fit_start_step"]) if "fit_start_step" in snap_meta.files else 0
            fe = int(snap_meta["fit_steps"])
            tot = int(snap_meta["total_steps"])
            pct = 100.0 * (fe - fs) / max(tot, 1)
            fit_resolved_str = (f"resolved: steps [{fs}, {fe})  "
                                f"of {tot} total  ({pct:.1f}% of training)")
        else:
            fit_resolved_str = ""
    except Exception:
        fit_resolved_str = ""

    if noise > 0:
        noise_html = (f"<code>NOISE_SIGMA = {noise:.3e}</code> "
                      "(rel. to trajectory std; baked into the snapshot cache)")
    else:
        noise_html = '<code>NOISE_SIGMA = 0.0</code> (clean snapshots)'

    out.append('<dl class="meta">')
    out.append(f"<dt>Label</dt><dd><code>{h(exp_label)}</code></dd>")
    out.append(f"<dt>Run time</dt><dd><code>{h(iso_time)}</code></dd>")
    out.append(f"<dt>Dataset</dt><dd><code>{h(str(getattr(C, 'DATASET', '—')).upper())}</code></dd>")
    out.append(f"<dt>Data cache</dt><dd><code>outputs/data/{h(dh)}/</code></dd>")
    out.append(f"<dt>Epochs</dt><dd><code>{h(str(getattr(C, 'EPOCHS', '—')))}</code></dd>")
    fit_dd = f"<code>{h(fit_knob_str)}</code>"
    if fit_resolved_str:
        fit_dd += f"<br><span class='note'>{h(fit_resolved_str)}</span>"
    out.append(f"<dt>Fit window</dt><dd>{fit_dd}</dd>")
    out.append(f"<dt>Snapshot noise</dt><dd>{noise_html}</dd>")
    out.append(f"<dt>Precision / repro tier</dt>"
               f"<dd><code>{h(str(getattr(C, 'PRECISION', 'float64')))}</code>"
               f" / <code>{h(str(getattr(C, 'REPRO_TIER', 'fast')))}</code></dd>")
    out.append("</dl>")

    # ----- 2. Layman intro -----
    dataset_name = str(getattr(C, "DATASET", "the dataset")).upper()
    n_classes = int(C.ARCH[-1]) if getattr(C, "ARCH", None) else None
    arch_str  = " → ".join(str(x) for x in getattr(C, "ARCH", []) or [])
    methods_in_run = ", ".join(_method_label(m) for m in methods)
    # Best-effort model-class name + parameter count. The training stack
    # builds whatever model.py exposes for the current run; if it ever
    # diverges from "an MLP", the description below still reads true.
    model_class_name = "neural network"
    n_params = None
    try:
        from .model import MLP as _ModelClass  # type: ignore
        model_class_name = _ModelClass.__name__
        try:
            _instance = _ModelClass(getattr(C, "ARCH", []))
            n_params  = sum(p.numel() for p in _instance.parameters())
        except Exception:
            pass
    except Exception:
        pass

    out.append("<h2>What this report shows</h2>")
    classes_hint = (f" ({n_classes} classes)" if n_classes else "")
    arch_hint    = ""
    if arch_str:
        arch_hint = (f" The classifier is a <code>{h(model_class_name)}</code> "
                     f"with layer widths <code>{h(arch_str)}</code>")
        if n_params:
            arch_hint += f" ({n_params:,} trainable parameters)"
        arch_hint += "."
    methods_hint = (f" Methods compared in this run: "
                    f"<code>{h(methods_in_run)}</code>." if methods_in_run else "")
    out.append(
        '<div class="description" style="font-style: normal;">'
        f"<p>We train a neural-network classifier on "
        f"<strong>{h(dataset_name)}</strong>{h(classes_hint)}.{arch_hint} "
        "While training, we record the network's full parameter vector "
        "at recorded snapshots, producing a trajectory through weight "
        "space. Nothing in the DMD pipeline below depends on the "
        "network architecture &mdash; swap the model and everything "
        "re-runs unchanged.</p>"
        "<p>DMD-family methods take the <strong>fit window</strong> "
        "portion of that trajectory and learn a low-rank linear model "
        "of how the weights evolve. We then ask each method to "
        "<em>extrapolate</em> the trajectory through the rest of "
        "training (the <strong>forecast region</strong>) without "
        "seeing those checkpoints during fitting."
        f"{methods_hint}</p>"
        "<p>At every recorded step we evaluate two networks on the "
        "<em>test</em> set: the <strong>real neural network</strong> "
        "(the genuine trained weights at that step) and the "
        "<strong>forecasted neural network</strong> (the DMD-predicted "
        "weights at the same step). Comparing the two tells us how "
        "faithfully each method captures real training dynamics.</p>"
        "<p><strong>How to read the plots.</strong> The dashed vertical "
        "line is the fit/forecast split. To the left of it the forecast "
        "was trained to match the real curve; to the right it has to "
        "predict without help. A good method tracks the real curve "
        "closely on <em>both</em> sides. In the eigenvalue plot, "
        "eigenvalues inside the unit circle ⇒ stable forecast; outside ⇒ "
        "the forecast eventually diverges.</p>"
        "</div>")
    out.append("<h3>Metric glossary</h3>")
    out.append('<p class="note">Every test-set metric in the tables and '
               "plots below is defined here. All accuracies / precisions / "
               "recalls / F1 scores live in [0, 1].</p>")
    out.append(_metric_glossary_html())

    # ----- 3. Real neural network headline -----
    actual_block = metrics.get("__actual__")
    if actual_block:
        out.append("<h2>Real neural network — final test performance</h2>")
        out.append(
            '<p class="note">The genuine trained classifier evaluated on the '
            f"{h(dataset_name)} test set at the <strong>last</strong> recorded "
            "training step. These are the reference numbers every forecast is "
            "compared against.</p>")
        out.append("<table>")
        out.append("<thead><tr>")
        for key, header in _TEST_METRICS:
            out.append(f"<th>{h(header)}</th>")
        out.append("</tr></thead><tbody><tr>")
        for key, _ in _TEST_METRICS:
            v = actual_block.get(f"{key}_final")
            out.append(f"<td class='num'>{h(_fmt_cell(key, v))}</td>")
        out.append("</tr></tbody></table>")

        cm = actual_block.get("cm_final")
        if _is_confusion_matrix(cm):
            out.append("<h3>Confusion matrix at the final step "
                       "<span class='note' style='font-weight:normal;'>"
                       "(rows = true class, columns = predicted class; "
                       "diagonal = correct, off-diagonal = mistakes)"
                       "</span></h3>")
            out.append(_render_confusion_table(cm, h))

    # ----- 4. Forecasted networks: final test performance per method -----
    has_pred = any("pred_loss_final" in metrics.get(mth, {}) for mth in methods)
    if has_pred:
        out.append("<h2>Forecasted neural networks — final test performance "
                   "by method</h2>")
        out.append(
            '<p class="note">For each method, we load the <em>forecasted</em> '
            "weights at the final training step into the same network and "
            "evaluate on the test set. The <code>Δ</code> column for loss / "
            "accuracy is <code>forecast − real</code>: a small magnitude "
            "(close to 0) means the forecast lands near the real network's "
            "performance. Loss <code>Δ</code> &gt; 0 ⇒ forecast under-performs; "
            "accuracy <code>Δ</code> &gt; 0 ⇒ forecast over-shoots.</p>")
        out.append("<table>")
        out.append("<thead><tr>")
        out.append("<th>Method</th>")
        out.append("<th>loss</th><th>Δloss</th>")
        out.append("<th>accuracy</th><th>Δacc</th>")
        out.append("<th>precision</th><th>recall</th><th>F1</th>")
        out.append("</tr></thead><tbody>")
        for method in methods:
            m = metrics.get(method, {})
            out.append("<tr>")
            out.append(f'<td class="method-cell"><code>{h(method)}</code></td>')
            for key in ("loss", "accuracy"):
                pf  = m.get(f"pred_{key}_final")
                gap = m.get(f"gap_{key}_final")
                out.append(f"<td class='num'>{h(_fmt_cell(key, pf))}</td>")
                out.append(f"<td class='num'>{h(_fmt_cell(f'gap_{key}', gap))}</td>")
            for key in ("precision", "recall", "f1"):
                pf = m.get(f"pred_{key}_final")
                out.append(f"<td class='num'>{h(_fmt_cell(key, pf))}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")

        # Per-method confusion matrix at the same (final) step. Laid out
        # in a responsive grid so they sit side-by-side on wide screens
        # and reflow to one-per-row on narrow ones. Compare against the
        # real network's confusion matrix in section 3 - the *pattern*
        # of off-diagonal mass is what reveals which classes the
        # forecast struggles with.
        pred_cms = [(m, metrics.get(m, {}).get("pred_cm_final"))
                    for m in methods]
        pred_cms = [(m, cm) for m, cm in pred_cms if _is_confusion_matrix(cm)]
        if pred_cms:
            out.append("<h3>Confusion matrix at the final step — per method "
                       "<span class='note' style='font-weight:normal;'>"
                       "(rows = true class, columns = predicted class). "
                       "Compare the off-diagonal pattern against the real "
                       "network above: which classes does each forecast "
                       "trip up on?</span></h3>")
            out.append('<div class="cm-strip">')
            for method, cm in pred_cms:
                out.append("<div>")
                out.append(f"<h4><code>{h(method)}</code> — "
                           f"{h(_method_label(method))}</h4>")
                out.append(_render_confusion_table(cm, h))
                out.append("</div>")
            out.append("</div>")

    # ----- 5. Operator stability + cost -----
    out.append("<h2>Operator stability &amp; runtime cost</h2>")
    out.append(
        '<p class="note">Each method fits a low-rank linear operator <code>A</code>; '
        "<strong>rank</strong> is its size, <strong>spectral ρ</strong> is the "
        "magnitude of its largest eigenvalue, and <strong>λ&gt;1</strong> counts "
        "eigenvalues outside the unit circle. ρ ≤ 1 and λ&gt;1 = 0 ⇒ the forecast "
        "is bounded; otherwise the forecast will grow exponentially. "
        "<strong>LM iters / stop</strong> show how the variable-projection "
        "Levenberg–Marquardt solver terminated for OptDMDc/cOptDMDc. "
        "<strong>Wall (s)</strong> is the analyse-time cost.</p>")
    out.append("<table>")
    out.append("<thead><tr><th>Method</th>")
    for _, header in _STABILITY_KEYS:
        out.append(f"<th>{h(header)}</th>")
    out.append("</tr></thead><tbody>")
    for method in methods:
        m = metrics.get(method, {})
        out.append("<tr>")
        out.append(f'<td class="method-cell"><code>{h(method)}</code></td>')
        for key, _ in _STABILITY_KEYS:
            v = m.get(key)
            cls = _classify_cell(key, v)
            out.append(f"<td class='{cls}'>{h(_fmt_cell(key, v))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")

    # Per-method unstable-eigenvalue list (only when n_outside > 0). LM
    # residuals appear inline below if available - one block per method
    # that has them.
    unstable_methods = [m for m in methods
                        if int(metrics.get(m, {}).get("n_outside", 0) or 0) > 0]
    lm_methods = [m for m in methods
                  if "lm_initial_residual" in metrics.get(m, {})]
    if unstable_methods or lm_methods:
        out.append("<h3>Per-method stability details</h3>")
        for method in unstable_methods:
            m = metrics[method]
            n_out = int(m.get("n_outside", 0))
            out.append(f"<h4><code>{h(method)}</code> — {n_out} "
                       "eigenvalue(s) outside the unit circle</h4>")
            out.append('<ul class="eigvals">')
            for ev in m.get("outside_top", [])[:10]:
                idx  = ev.get("index", "?")
                re_v = ev.get("real",      float("nan"))
                im_v = ev.get("imag",      float("nan"))
                mag  = ev.get("magnitude", float("nan"))
                exc  = ev.get("excess",    float("nan"))
                out.append(
                    f"<li>λ<sub>{h(str(idx))}</sub> = {re_v:+.6f}{im_v:+.6f}j  "
                    f"|λ|={mag:.6f}  excess={exc:+.3e}</li>")
            out.append("</ul>")
        for method in lm_methods:
            m  = metrics[method]
            ri = m.get("lm_initial_residual", float("nan"))
            rf = m.get("lm_final_residual",   float("nan"))
            it = m.get("lm_iters", "?")
            rs_ = m.get("lm_restarts", 0)
            out.append(
                '<div class="lm-note">'
                f"<code>{h(method)}</code> LM: residual "
                f"{_fmt_cell('loss', ri)} → {_fmt_cell('loss', rf)}  "
                f"({h(str(it))} iters, {h(str(rs_))} restarts)</div>")

    # ----- 6. Rank scan diagnostics -----
    rs = metrics.get("__rank_scan__")
    if rs:
        out.append("<h2>Rank scan (auto-selected LM rank)</h2>")
        out.append(
            '<p class="note">Before fitting OptDMDc / cOptDMDc, we sweep '
            "a list of candidate ranks and pick the one whose held-out "
            "forecast has the lowest relative L2 error. The chosen rank "
            "is patched into <code>METHOD_PARAMS</code> for both LM methods. "
            "The shaded row below is the winner.</p>")
        out.append('<dl class="meta">')
        out.append(f"<dt>Best rank</dt><dd><code>{h(str(rs.get('best_rank')))}</code></dd>")
        bve = rs.get("best_val_err")
        out.append(f"<dt>Best val L2 error</dt><dd>{h(_fmt_cell('val_err', bve))}</dd>")
        out.append(f"<dt>Patience</dt><dd>{h(str(rs.get('patience')))}</dd>")
        out.append(f"<dt>Validation fraction</dt><dd>{h(str(rs.get('val_frac')))}</dd>")
        out.append(f"<dt>Min relative improvement</dt><dd>{h(str(rs.get('min_improv')))}</dd>")
        out.append("</dl>")

        history = rs.get("history", []) or []
        if history:
            best_rank = rs.get("best_rank")
            out.append("<table>")
            out.append("<thead><tr><th>Candidate rank</th>"
                       "<th>Validation rel. L2 error</th></tr></thead><tbody>")
            for entry in history:
                r_ = entry.get("rank")
                e  = entry.get("val_rel_err")
                cls = ' class="best-rank"' if r_ == best_rank else ""
                out.append(
                    f"<tr{cls}>"
                    f"<td class='num'><code>{h(str(r_))}</code></td>"
                    f"<td class='num'>{h(_fmt_cell('val_err', e))}</td>"
                    "</tr>")
            out.append("</tbody></table>")

    # ----- 7. Plots -----
    out.append("<h2>Plots</h2>")

    # Cross-method overlay first: answers "which method tracks best?"
    # at a glance, before drilling into per-method detail below.
    overlay = exp_dir() / "plots" / "overlay" / "combined.png"
    if overlay.exists():
        out.append("<h3>Cross-method overlay</h3>")
        out.append(
            '<p class="note">One panel per network: the real one at the top, '
            "then each method's forecast below it. Inside every panel, four "
            "classification metrics (accuracy / precision / recall / F1, left "
            "axis, in [0, 1]) plus test loss (right axis, dashed) are plotted "
            "against training step. Compare panels to see which method most "
            "closely reproduces the real network's curve shape.</p>")
        b64 = base64.b64encode(overlay.read_bytes()).decode("ascii")
        out.append('<section class="method-plots"><div class="plot">')
        out.append(f'<img src="data:image/png;base64,{b64}" '
                   'alt="cross-method overlay">')
        out.append("</div></section>")

    # Per-method PNG grid. Skip 'parameter-vector accuracy' for users
    # who only ran the cls/loss subset (file simply won't exist).
    if methods:
        out.append("<h3>Per-method plots</h3>")
        out.append(
            '<p class="note">Each section drills into one DMD method: '
            "real vs forecasted curves over training steps, and the "
            "spectrum of the linear operator it learned.</p>")
    for method in methods:
        out.append('<section class="method-plots">')
        out.append(f"<h4><code>{h(method)}</code> — {h(_method_label(method))}</h4>")
        for kind in ("loss", "classification", "accuracy", "eigenvalues"):
            p = exp_dir() / "plots" / method / f"{kind}.png"
            if not p.exists():
                continue
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            out.append('<div class="plot">')
            out.append(
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{h(method)} {h(kind)}">')
            cap = _PLOT_CAPTIONS.get(kind, kind)
            out.append(f'<div class="plot-caption">{h(cap)}</div>')
            out.append("</div>")
        out.append("</section>")

    # ----- 8. Full configuration snapshot (collapsible) -----
    out.append("<h2>Full configuration</h2>")
    out.append('<p class="note">Every tunable knob at the moment this '
               "experiment was launched. The training-relevant subset "
               "(<code>DATASET</code>, <code>ARCH</code>, <code>EPOCHS</code>, "
               "<code>LR</code>, <code>BATCH_SIZE</code>, <code>SEED</code>, "
               "<code>NOISE_SIGMA</code>, fit window) determines the "
               "<code>data_hash</code>; everything else is an analysis-time "
               "choice.</p>")
    out.append(_render_config_dump(h))

    out.append("</main></body></html>")

    summary_path().write_text("\n".join(out), encoding="utf-8")
    size_kb = summary_path().stat().st_size / 1024
    log("ok",
        f"experiments.write_summary: wrote {summary_path()}  ({size_kb:.0f} KB)")

    # Drop legacy markdown reports if any prior version of the project
    # left them next to the new HTML report.
    for legacy in (exp_dir() / "summary.md",):
        if legacy.exists():
            legacy.unlink()
            log("info", f"experiments.write_summary: removed stale {legacy}")
