"""
Recorder + on-disk snapshot schema.

Captures the parameter trajectory + control inputs during training so
DMDc-family analyses can be fit afterwards. Two recording cadences:

    fit window      [fit_start_step, fit_end_step)   every fit_every steps
    elsewhere       [0, fit_start_step)
                  + [fit_end_step,   total_steps)    every forecast_every steps

The fit window can begin at any step (not just 0). FIT_RANGE = (start,
end) in config.py drives this; FIT_RANGE = None keeps the historical
behaviour (start=0, end = FIT_FRAC * total_steps).

The control vector u_k is recorded at *every* gradient step regardless
of whether a snapshot is taken, because the DMDc forecast iterates step
by step in the fit-time-unit (= fit_every gradient steps) and needs u_k
for every step in the forecast region too.

# .npz schema written by save():
    X                (n, m)               state snapshots, ordered by step
    U                (q, total_steps - 1) full per-step control sequence
                                          (drives x_k -> x_{k+1} for every k)
    steps            (m,)                 gradient step at which each snapshot
                                          was taken
    fit_start_step   scalar int           gradient step where fit window begins (>= 0)
    fit_steps        scalar int           gradient step where fit window ends (= old "fit_steps")
    fit_start_idx    scalar int           snapshot index where fit window begins
    fit_split        scalar int           snapshot index where fit window ends (one past last)
    total_steps      scalar int           total gradient steps trained
    fit_every        scalar int           dense (in-fit) snapshot cadence
    forecast_every   scalar int           sparse (out-of-fit) snapshot cadence
"""

import numpy as np

from .params import flatten_params


def eval_indices(steps: np.ndarray, every: int) -> np.ndarray:
    # Indices into a snapshot array where the recorded gradient step is a
    # multiple of `every`. Used by plot_loss / plot_accuracy to subsample
    # the dense fit-region snapshots down to the same per-N grid as the
    # forecast region.
    return np.where(steps % every == 0)[0]


class Recorder:
    def __init__(self, *, fit_every: int, forecast_every: int,
                 fit_start_step: int, fit_end_step: int,
                 total_steps: int, control_fn):
        # fit_every       : record every N steps inside [fit_start_step, fit_end_step)
        # forecast_every  : record every N steps outside that window
        # fit_start_step  : gradient-step boundary; first fit-region step
        # fit_end_step    : gradient-step boundary; first forecast-region step
        # total_steps     : total training steps; controls list will have
        #                   length total_steps - 1
        # control_fn      : (optimizer, step) -> 1-D iterable of floats
        if not (0 <= fit_start_step < fit_end_step <= total_steps):
            raise ValueError(
                f"invalid fit window: fit_start_step={fit_start_step}  "
                f"fit_end_step={fit_end_step}  total_steps={total_steps}")
        self.fit_every       = fit_every
        self.forecast_every  = forecast_every
        self.fit_start_step  = fit_start_step
        self.fit_end_step    = fit_end_step
        self.total_steps     = total_steps
        self.control_fn      = control_fn

        self.X: list[np.ndarray] = []
        self.U: list[list[float]] = []
        self.steps: list[int] = []
        self._k = 0

    def _is_snapshot_step(self, k: int) -> bool:
        if self.fit_start_step <= k < self.fit_end_step:
            return ((k - self.fit_start_step) % self.fit_every) == 0
        # outside fit window: align sparse cadence to global step 0 so
        # plot_loss / plot_accuracy subsampling (eval_indices) works
        # uniformly on either side of the fit window
        return (k % self.forecast_every) == 0

    def step(self, model, optimizer) -> None:
        # Call once per gradient step (after opt.step + sched.step).
        # Records the parameter vector iff this step is on a snapshot grid;
        # always records the control vector u_k (so U is dense across the run).
        k = self._k
        if self._is_snapshot_step(k):
            self.X.append(flatten_params(model))
            self.steps.append(k)
        self.U.append(list(self.control_fn(optimizer, k)))
        self._k += 1

    @staticmethod
    def load(path: str) -> dict:
        # Inverse of save(). Returns a dict with all arrays + scalar
        # metadata. Backwards-compat: schemas written before FIT_RANGE
        # was added (no fit_start_step / fit_start_idx) default to 0.
        npz = np.load(path)
        keys = set(npz.files)
        out = {
            "X":              npz["X"],
            "U":              npz["U"],
            "steps":          npz["steps"],
            "fit_split":      int(npz["fit_split"]),
            "fit_steps":      int(npz["fit_steps"]),
            "total_steps":    int(npz["total_steps"]),
            "fit_every":      int(npz["fit_every"]),
            "forecast_every": int(npz["forecast_every"]),
            "fit_start_step": int(npz["fit_start_step"]) if "fit_start_step" in keys else 0,
            "fit_start_idx":  int(npz["fit_start_idx"])  if "fit_start_idx"  in keys else 0,
        }
        return out

    def save(self, path: str) -> None:
        # X is the snapshot stack; U[:total_steps-1] gives the per-step
        # control sequence used to roll the network from x_0 to x_{T-1}.
        # The very last element of U (recorded after the final opt.step) has
        # no successor and is dropped.
        X = np.stack(self.X, axis=1)                          # (n, m)
        U = np.asarray(self.U[:self.total_steps - 1]).T       # (q, T-1)
        steps = np.asarray(self.steps, dtype=np.int64)
        fit_start_idx = int(np.searchsorted(steps, self.fit_start_step,
                                            side="left"))
        fit_split     = int(np.searchsorted(steps, self.fit_end_step,
                                            side="left"))
        np.savez(
            path,
            X=X.astype(np.float32),
            U=U.astype(np.float32),
            steps=steps,
            fit_start_step=np.int64(self.fit_start_step),
            fit_steps=np.int64(self.fit_end_step),
            fit_start_idx=np.int64(fit_start_idx),
            fit_split=np.int64(fit_split),
            total_steps=np.int64(self.total_steps),
            fit_every=np.int64(self.fit_every),
            forecast_every=np.int64(self.forecast_every),
        )
