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
