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
import inspect
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
    "BATCH_SIZE",
    "LR",
    "LR_MIN",
    "LOSS",
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

def train_config_dict() -> dict:
    """Subset of config used to derive the data_hash.

    Special handling for the fit-window selector:
        FIT_RANGE wins at runtime when it is not None (FIT_FRAC is
        ignored). To match that semantics, we hash ONLY the active
        selector - so toggling FIT_FRAC while FIT_RANGE is set, or
        toggling FIT_RANGE while it stays None, does NOT invalidate
        the cache. Whichever value would actually drive the recorder
        is the only one that participates in the hash.
    """
    fields: dict[str, Any] = {}
    for k in TRAIN_FIELDS:
        if k in ("FIT_FRAC", "FIT_RANGE"):
            continue   # handled below as the active selector
        v = getattr(C, k, None)
        if isinstance(v, list):
            v = tuple(v)
        fields[k] = v

    fr = getattr(C, "FIT_RANGE", None)
    if fr is not None:
        fields["FIT_RANGE"] = tuple(fr) if isinstance(fr, (list, tuple)) else fr
    else:
        fields["FIT_FRAC"] = float(getattr(C, "FIT_FRAC", 0.5))

    fields["control_fn:source"] = inspect.getsource(C.control_fn)
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
    if callable(getattr(C, "control_fn", None)):
        out["control_fn:source"] = inspect.getsource(C.control_fn)
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

_SUMMARY_KEYS = [
    ("rank",                "Rank"),
    ("pod_rank",            "POD rank"),
    ("spectral_radius",     "Spectral ρ"),
    ("n_outside",           "λ>1"),
    ("in_sample_dacc_pct",  "In Δacc%"),
    ("out_sample_dacc_pct", "Out Δacc%"),
    ("in_sample_dloss",     "In Δloss"),
    ("out_sample_dloss",    "Out Δloss"),
    ("lm_iters",            "LM iters"),
    ("lm_stop_reason",      "LM stop"),
    ("wall_seconds",        "Wall (s)"),
]

_FORECAST_KEYS = [
    ("forecast_loss_actual_final", "actual final L"),
    ("forecast_loss_pred_final",   "pred final L"),
    ("forecast_loss_actual_min",   "actual min L"),
    ("forecast_loss_pred_min",     "pred min L"),
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
        if "spectral" in key or "loss" in key:
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
""".strip()


def write_summary() -> None:
    """Aggregate metrics.json + manifest into a self-contained HTML
    report at <exp_dir>/summary.html. PNG plots are inlined as base64
    data URLs so the file renders correctly in any browser regardless
    of where it's moved."""
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
    label     = manifest.get("label", "default")
    desc      = manifest.get("description", "")
    iso_time  = manifest.get("iso_time", "")
    dh        = manifest.get("data_hash", data_hash())

    methods_cfg = list(getattr(C, "METHODS", tuple(metrics.keys())))
    # Method-row entries: ignore any "__meta__" keys (e.g. __rank_scan__),
    # which are rendered in their own dedicated sections.
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

    # header
    out.append(f"<h1>Experiment: <code>{h(eid)}</code></h1>")
    if desc:
        out.append(f'<div class="description">{h(desc)}</div>')

    # Surface the training-relevant config knobs that change how the
    # numbers below should be interpreted (noise level, fit window,
    # epochs, precision, repro tier). Each of these participates in
    # data_hash, so they're stable for the lifetime of the experiment;
    # we read them from the active config module.
    noise = float(getattr(C, "NOISE_SIGMA", 0.0) or 0.0)
    fit_range = getattr(C, "FIT_RANGE", None)
    fit_frac  = getattr(C, "FIT_FRAC", None)
    if fit_range is not None:
        fit_window_str = (f"FIT_RANGE = ({int(fit_range[0])}, "
                          f"{int(fit_range[1])})")
    else:
        fit_window_str = (f"FIT_FRAC = {fit_frac:.3f}"
                          if fit_frac is not None else "—")

    if noise > 0:
        noise_html = (f"<code>NOISE_SIGMA = {noise:.3e}</code> "
                      "(rel. to trajectory std; noise is BAKED into "
                      "this experiment's snapshot cache)")
    else:
        noise_html = '<code>NOISE_SIGMA = 0.0</code> (clean snapshots)'

    out.append('<dl class="meta">')
    out.append(f"<dt>Label</dt><dd><code>{h(label)}</code></dd>")
    out.append(f"<dt>Run time</dt><dd><code>{h(iso_time)}</code></dd>")
    out.append(f"<dt>Data</dt><dd><code>outputs/data/{h(dh)}/</code></dd>")
    out.append(f"<dt>Methods</dt><dd><code>{h(', '.join(methods))}</code></dd>")
    out.append(f"<dt>Snapshot noise</dt><dd>{noise_html}</dd>")
    out.append(f"<dt>Fit window</dt><dd><code>{h(fit_window_str)}</code></dd>")
    out.append(f"<dt>Epochs</dt><dd><code>{h(str(getattr(C, 'EPOCHS', '—')))}</code></dd>")
    out.append(f"<dt>Precision</dt><dd><code>{h(str(getattr(C, 'PRECISION', 'float64')))}</code></dd>")
    out.append(f"<dt>Repro tier</dt><dd><code>{h(str(getattr(C, 'REPRO_TIER', 'fast')))}</code></dd>")
    out.append("</dl>")

    # --- main metrics table ---
    out.append("<h2>Metrics</h2>")
    out.append("<table>")
    out.append("<thead><tr>")
    out.append("<th>Method</th>")
    for _, header in _SUMMARY_KEYS:
        out.append(f"<th>{h(header)}</th>")
    out.append("</tr></thead><tbody>")
    for method in methods:
        m = metrics.get(method, {})
        out.append("<tr>")
        out.append(f'<td class="method-cell"><code>{h(method)}</code></td>')
        for key, _ in _SUMMARY_KEYS:
            v = m.get(key)
            cls = _classify_cell(key, v)
            out.append(f'<td class="{cls}">{h(_fmt_cell(key, v))}</td>')
        out.append("</tr>")
    out.append("</tbody></table>")

    # --- forecast-region final / min ---
    has_forecast = any(any(k in metrics.get(m, {}) for k, _ in _FORECAST_KEYS)
                       for m in methods)
    if has_forecast:
        out.append("<h2>Forecast-region final / min</h2>")
        out.append('<p class="note">Predicted vs actual at the <strong>last</strong> '
                   "forecast snapshot, and the <strong>minimum</strong> over the "
                   "forecast region. Useful for spotting whether a method's "
                   "prediction stays near the true loss / accuracy or drifts.</p>")
        out.append("<table>")
        out.append("<thead><tr><th>Method</th>")
        for _, header in _FORECAST_KEYS:
            out.append(f"<th>{h(header)}</th>")
        out.append("</tr></thead><tbody>")
        for method in methods:
            m = metrics.get(method, {})
            out.append("<tr>")
            out.append(f'<td class="method-cell"><code>{h(method)}</code></td>')
            for key, _ in _FORECAST_KEYS:
                v = m.get(key)
                cls = _classify_cell(key, v)
                out.append(f'<td class="{cls}">{h(_fmt_cell(key, v))}</td>')
            out.append("</tr>")
        out.append("</tbody></table>")

    # --- eigenvalue stability ---
    out.append("<h2>Eigenvalue stability</h2>")
    for method in methods:
        m = metrics.get(method, {})
        n_out = int(m.get("n_outside", 0) or 0)
        sr    = m.get("spectral_radius", float("nan"))
        if n_out:
            out.append(f"<h3><code>{h(method)}</code> — {n_out} eigenvalue(s) "
                       "outside unit circle</h3>")
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
        else:
            sr_str = f"{sr:.6f}" if isinstance(sr, (int, float)) else str(sr)
            out.append('<div class="stable-note">'
                       f'<code>{h(method)}</code>: all eigenvalues inside unit circle '
                       f'(spectral radius {sr_str}).</div>')

    # --- rank scan diagnostics (if the auto scan ran) ---
    rs = metrics.get("__rank_scan__")
    if rs:
        out.append("<h2>LM rank scan</h2>")
        out.append('<p class="note">Held-out forecast scan over candidate '
                   "ranks; lowest validation L2 error wins. The chosen rank "
                   "was patched into <code>METHOD_PARAMS</code> for both "
                   "<code>optdmdc</code> and <code>coptdmdc</code>.</p>")
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
                r = entry.get("rank")
                e = entry.get("val_rel_err")
                cls = ' class="best-rank"' if r == best_rank else ""
                out.append(
                    f"<tr{cls}>"
                    f"<td class='num'><code>{h(str(r))}</code></td>"
                    f"<td class='num'>{h(_fmt_cell('val_err', e))}</td>"
                    "</tr>")
            out.append("</tbody></table>")

    # --- LM convergence ---
    lm_methods = [m for m in methods
                  if "lm_initial_residual" in metrics.get(m, {})]
    if lm_methods:
        out.append("<h2>LM convergence</h2>")
        for method in lm_methods:
            m = metrics[method]
            ri = m.get("lm_initial_residual", "?")
            rf = m.get("lm_final_residual", "?")
            it = m.get("lm_iters", "?")
            rs = m.get("lm_restarts", 0)
            out.append('<div class="lm-note">'
                       f"<code>{h(method)}</code>: residual {ri:.6e} → {rf:.6e}  "
                       f"({it} iters, {rs} restarts)</div>")

    # --- plots (PNGs embedded as base64 data URLs) ---
    out.append("<h2>Plots</h2>")
    for method in methods:
        out.append('<section class="method-plots">')
        out.append(f"<h3><code>{h(method)}</code></h3>")
        for kind in ("loss", "accuracy", "eigenvalues"):
            p = exp_dir() / "plots" / method / f"{kind}.png"
            if not p.exists():
                continue
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            out.append('<div class="plot">')
            out.append(
                f'<img src="data:image/png;base64,{b64}" '
                f'alt="{h(method)} {h(kind)}">')
            out.append(f'<div class="plot-caption">{h(method)} — {h(kind)}</div>')
            out.append("</div>")
        out.append("</section>")

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
