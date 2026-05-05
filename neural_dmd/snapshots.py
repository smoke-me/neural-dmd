import numpy as np

from .params import flatten_params


def eval_indices(steps: np.ndarray, every: int) -> np.ndarray:
    # Indices into a snapshot array where the recorded gradient step is a
    # multiple of `every`. Used by plot_loss.py / plot_accuracy.py to
    # subsample the dense fit-region snapshots down to the same per-50
    # grid as the forecast region.
    return np.where(steps % every == 0)[0]

# Captures the parameter trajectory + control inputs during training so DMDc
# can be fit afterwards. Two recording cadences:
#
#   * fit region      [0, fit_steps)            - record every fit_every steps
#                                                 (default every step)
#   * forecast region [fit_steps, total_steps)  - record every forecast_every
#                                                 steps (sparse comparison
#                                                 grid for plotting)
#
# The control vector u_k is recorded at *every* gradient step regardless of
# whether a snapshot is taken, because the DMDc forecast iterates step by
# step in the fit-time-unit (= fit_every gradient steps) and needs u_k for
# every step in the forecast region too.
#
# .npz schema written by save():
#   X              (n, m)                  state snapshots, ordered by step
#   U              (q, total_steps - 1)    full per-step control sequence
#                                          (drives x_k -> x_{k+1} for every k)
#   steps          (m,)                    gradient step at which each snapshot
#                                          was taken
#   fit_split      scalar int              number of fit-region snapshots
#                                          (= len of fit data fed to DMDc)
#   fit_steps      scalar int              gradient-step boundary between fit
#                                          and forecast regions
#   total_steps    scalar int              total gradient steps trained
#   fit_every      scalar int              fit-region snapshot cadence
#   forecast_every scalar int              forecast-region snapshot cadence


class Recorder:
    def __init__(self, *, fit_every: int, forecast_every: int,
                 fit_steps: int, total_steps: int, control_fn):
        # fit_every       : record every N steps inside [0, fit_steps)
        # forecast_every  : record every N steps inside [fit_steps, total_steps)
        # fit_steps       : gradient-step boundary; first forecast-region step
        # total_steps     : total training steps; controls list will have
        #                   length total_steps - 1
        # control_fn      : (optimizer, step) -> 1-D iterable of floats
        self.fit_every = fit_every
        self.forecast_every = forecast_every
        self.fit_steps = fit_steps
        self.total_steps = total_steps
        self.control_fn = control_fn

        self.X: list[np.ndarray] = []
        self.U: list[list[float]] = []
        self.steps: list[int] = []
        self._k = 0

    def _is_snapshot_step(self, k: int) -> bool:
        if k < self.fit_steps:
            return (k % self.fit_every) == 0
        # forecast region: align to fit boundary (k - fit_steps multiples)
        return ((k - self.fit_steps) % self.forecast_every) == 0

    def step(self, model, optimizer) -> None:
        # Call once per gradient step (after opt.step + sched.step). Records
        # the parameter vector if this step is on a snapshot grid; always
        # records the control vector u_k (so U is dense across the run).
        k = self._k
        if self._is_snapshot_step(k):
            self.X.append(flatten_params(model))
            self.steps.append(k)
        self.U.append(list(self.control_fn(optimizer, k)))
        self._k += 1

    @staticmethod
    def load(path: str) -> dict:
        # Inverse of save(). Returns a dict with all arrays + scalar metadata.
        # Used by analyze.py and the plot_*.py scripts.
        npz = np.load(path)
        return {
            "X":              npz["X"],
            "U":              npz["U"],
            "steps":          npz["steps"],
            "fit_split":      int(npz["fit_split"]),
            "fit_steps":      int(npz["fit_steps"]),
            "total_steps":    int(npz["total_steps"]),
            "fit_every":      int(npz["fit_every"]),
            "forecast_every": int(npz["forecast_every"]),
        }

    def save(self, path: str) -> None:
        # X is the snapshot stack; U[:total_steps-1] gives the per-step
        # control sequence used to roll the network from x_0 to x_{T-1}.
        # The very last element of U (recorded after the final opt.step) has
        # no successor and is dropped.
        X = np.stack(self.X, axis=1)                          # (n, m)
        U = np.asarray(self.U[:self.total_steps - 1]).T       # (q, T-1)
        steps = np.asarray(self.steps, dtype=np.int64)
        fit_split = int(np.searchsorted(steps, self.fit_steps))
        np.savez(
            path,
            X=X.astype(np.float32),
            U=U.astype(np.float32),
            steps=steps,
            fit_split=np.int64(fit_split),
            fit_steps=np.int64(self.fit_steps),
            total_steps=np.int64(self.total_steps),
            fit_every=np.int64(self.fit_every),
            forecast_every=np.int64(self.forecast_every),
        )
