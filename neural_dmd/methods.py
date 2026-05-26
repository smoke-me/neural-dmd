"""
Method registry. Single source of truth for which DMDc-family methods
are available, what their `run` callable is, and how their results are
labelled.

Adding a new method:
    1. Implement neural_dmd/<name>.py with a `run(snap, **kwargs)`
       returning a dict containing at least 'X_pred', 'A',
       'eigenvalues' (or just 'A'), 'rank', 'fit_split'.
    2. Add an entry to METHODS below.
    3. (Optional) Add hyperparameters to config.METHOD_PARAMS[<name>].
    4. Add the method to config.METHODS to enable it for an experiment.

The orchestrator (scripts/run_exp.py) and the unified analyze runner
(scripts/_run_method.py) consume this registry.
"""

from __future__ import annotations

from . import dmdc, sdmdc, optdmdc, coptdmdc, optdmdc_direct, coptdmdc_direct


METHODS: dict[str, dict] = {
    "dmdc": {
        "label":   "DMDc",
        "run":     dmdc.run,
        "has_lm":  False,
        "blurb":   "Proctor-Brunton-Kutz DMDc, no constraint, no optimisation.",
    },
    "sdmdc": {
        "label":   "sDMDc",
        "run":     sdmdc.run,
        "has_lm":  False,
        "blurb":   "DMDc + one-shot radial-projection clipping of unstable "
                   "eigenvalues. No LM.",
    },
    "optdmdc": {
        "label":   "OptDMDc (log)",
        "run":     optdmdc.run,
        "has_lm":  True,
        "blurb":   "Askham-Kutz Optimized DMD with control. Variable "
                   "projection LM in continuous-time γ = log(λ)/dt. "
                   "Fragile when warm-start has tiny |λ|.",
    },
    "coptdmdc": {
        "label":   "cOptDMDc (log)",
        "run":     coptdmdc.run,
        "has_lm":  True,
        "blurb":   "Constrained OptDMDc (log form). Re(γ) ≤ 0 enforced via "
                   "initial radial projection + linear-inequality LM step.",
    },
    "optdmdc_direct": {
        "label":   "OptDMDc (direct λ)",
        "run":     optdmdc_direct.run,
        "has_lm":  True,
        "blurb":   "Direct-λ OptDMDc. Same VarPro LM, parametrized by the "
                   "discrete-time eigenvalue. Forward-recurrence basis - "
                   "no log/exp, no tiny-|λ| overflow.",
    },
    "coptdmdc_direct": {
        "label":   "cOptDMDc (direct λ)",
        "run":     coptdmdc_direct.run,
        "has_lm":  True,
        "blurb":   "Constrained direct-λ OptDMDc. Stability enforced by "
                   "radial projection onto the unit disk after each LM "
                   "step (non-linear constraint, projection-based).",
    },
}


def get(name: str) -> dict:
    if name not in METHODS:
        raise KeyError(
            f"unknown method '{name}'. Registered: {list(METHODS)}")
    return METHODS[name]
