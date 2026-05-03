import numpy as np

from params import flatten_params

# Captures the parameter trajectory + control inputs during training so DMDc
# can be fit afterwards. Saved on disk as a single .npz file:
#   X     (n_params, m)        state snapshots x_0 .. x_{m-1}
#   U     (n_controls, m-1)    control inputs  u_0 .. u_{m-2}
#   steps (m,)                 training step index of each snapshot
#
# The control variable is whatever `control_fn(optimizer, step)` returns,
# so this module is agnostic to what u_k actually represents.


class Recorder:
    def __init__(self, every: int, control_fn):
        # `every`      : capture every N gradient steps (DMDc time = N steps)
        # `control_fn` : (optimizer, step) -> 1-D iterable of floats; this is
        #                the control vector u_k recorded with each snapshot.
        self.every = every
        self.control_fn = control_fn
        self.X: list[np.ndarray] = []
        self.U: list[list[float]] = []
        self.steps: list[int] = []
        self._k = 0

    def step(self, model, optimizer) -> None:
        # Call once per gradient step. Records on multiples of `every`.
        if self._k % self.every == 0:
            self.X.append(flatten_params(model))
            self.U.append(list(self.control_fn(optimizer, self._k)))
            self.steps.append(self._k)
        self._k += 1

    def save(self, path: str) -> None:
        # X has m snapshots; U[:-1] gives the m-1 controls that drove
        # x_k -> x_{k+1}. The last recorded U has no successor and is dropped.
        X = np.stack(self.X, axis=1)              # (n, m)
        U = np.asarray(self.U[:-1]).T             # (n_controls, m-1)
        steps = np.asarray(self.steps, dtype=np.int64)
        np.savez(path,
                 X=X.astype(np.float32),
                 U=U.astype(np.float32),
                 steps=steps)
