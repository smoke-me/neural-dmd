# Tiny colored console logger. Zero non-stdlib deps.
# Tags are short words so log lines stay readable both for humans and LLMs:
#   info  - neutral status (cyan)
#   ok    - success / metric you care about (green)
#   warn  - non-fatal anomaly (yellow)
#   err   - failure (red)

import time

# ANSI color codes mapped to each tag.
_C = {
    "info": "\033[36m",
    "ok":   "\033[32m",
    "warn": "\033[33m",
    "err":  "\033[31m",
}
_R = "\033[0m"  # reset color


def log(tag: str, msg: str) -> None:
    # Print "<colored TAG>  <message>". Tag is upper-cased and padded to 4
    # chars so columns line up across log lines.
    print(f"{_C[tag]}{tag.upper():<4}{_R}  {msg}", flush=True)


def banner(label: str, **fields) -> None:
    # One-line stage banner: "STAGE  label  k1=v1  k2=v2 ...". Used at the
    # start of each major step (training, DMDc fit, evaluation, plotting)
    # so the log reads like a journal of where the pipeline currently is.
    pieces = "  ".join(f"{k}={v}" for k, v in fields.items())
    log("info", f"== {label}  {pieces}".rstrip())


def progress(label: str, current: int, total: int, t_start: float, *,
             every_pct: float = 25.0) -> None:
    # Emit a progress line at every-Nth-percent milestone (default 25%) and
    # always at completion. Includes elapsed wall time and a linear ETA.
    # Call AFTER unit `current` is done.
    if total <= 0:
        return
    pct      = 100.0 * current / total
    prev_pct = 100.0 * (current - 1) / total if current > 0 else 0.0
    crossed  = int(pct / every_pct) > int(prev_pct / every_pct)
    if not (crossed or current == total):
        return
    elapsed = time.time() - t_start
    eta     = elapsed * (total - current) / current if current > 0 else 0.0
    log("info",
        f"  {label}: {current}/{total} ({pct:5.1f}%)  "
        f"elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")


def log_spectrum(label, sigma, top_n: int = 5):
    # Pretty-print a singular-value spectrum: top-N values, the ranks
    # required to capture 90 / 99 / 99.9% of cumulative L2 energy
    # (sum sigma_i^2), and the condition number sigma_0 / sigma_min.
    # Used by dmdc / optdmdc / coptdmdc kernels so the user can see how
    # sharply the trajectory's variance is concentrated in the top modes
    # (-> how lossy a low-rank truncation will be).
    import numpy as np  # noqa: WPS433  (deferred import to keep log.py stdlib-only otherwise)
    s = np.asarray(sigma, dtype=float)
    n = len(s)
    energy = s ** 2
    total  = energy.sum()
    if total <= 0.0:
        log("info", f"{label}: empty spectrum")
        return
    cum = np.cumsum(energy) / total
    def rank_for(thr):
        return int(np.searchsorted(cum, thr) + 1)
    k90  = rank_for(0.90)
    k99  = rank_for(0.99)
    k999 = rank_for(0.999)
    head = "  ".join(f"sigma_{i}={s[i]:.3e}" for i in range(min(top_n, n)))
    cond = s[0] / s[-1] if s[-1] > 0 else float("inf")
    log("info",
        f"{label}: n={n}  {head}  sigma_min={s[-1]:.3e}  "
        f"cond={cond:.2e}")
    log("info",
        f"{label}: rank for  90%={k90}  99%={k99}  99.9%={k999}  "
        f"100%={n}")
