import numpy as np

# Dynamic Mode Decomposition with control (DMDc).
#
# Fits a linear discrete-time model
#       x_{k+1} = A x_k + B u_k
# from snapshot data (X, U) and forecasts the trajectory forward.
#
# State dimension n is enormous (= number of NN parameters, ~10^5), while
# the number of snapshots m is small (~10^2). We never form A in R^{n*n}.
# Instead we project onto a low-dimensional POD basis discovered from the
# data (Proctor, Brunton, Kutz - SIAM J. Appl. Dyn. Syst. 2016).
#
# Algorithm (unknown B):
#   1. Build augmented matrix Omega = [X1; U]   (here X1 = X[:, :-1]).
#   2. SVD:  Omega = U_o S_o V_o^T   (truncate to rank r).
#   3. SVD:  X2     = U_x S_x V_x^T  (truncate to rank p; p <= r).
#   4. Split U_o row-wise into U_o1 (state part) and U_o2 (control part).
#   5. Reduced operators (act on y = U_x^T x):
#        A_red = U_x^T  X2  V_o  S_o^{-1}  U_o1^T  U_x   (p x p)
#        B_red = U_x^T  X2  V_o  S_o^{-1}  U_o2^T        (p x q)
#   6. Forecast in reduced space, lift back with U_x.


def fit_dmdc(X: np.ndarray, U: np.ndarray, rank: int | None = None) -> dict:
    # X: (n, m) state snapshots
    # U: (q, m-1) control inputs
    # rank: truncation for the reduced model (None -> use all available modes)
    n, m = X.shape
    q = U.shape[0]
    assert U.shape[1] == m - 1, "need m-1 controls for m snapshots"

    X1 = X[:, :-1]                          # (n, m-1)
    X2 = X[:, 1:]                           # (n, m-1)

    # 1-2. SVD of the input subspace.
    Omega = np.vstack([X1, U])              # (n+q, m-1)
    Uo, So, Vto = np.linalg.svd(Omega, full_matrices=False)
    r = len(So) if rank is None else min(rank, len(So))
    Uo, So, Vto = Uo[:, :r], So[:r], Vto[:r, :]

    # 3. SVD of the output subspace. Canonical Proctor DMDc uses X2 here,
    #    but that excludes x_0 from the basis. We use the full X so x_0
    #    lies exactly in span(Ux). We also do NOT cap p at r: input
    #    subspace rank is at most m-1 (Omega has m-1 columns), but we want
    #    the output basis to span all m snapshots so every recorded x_k is
    #    perfectly representable. Otherwise the smallest mode of X gets
    #    truncated and in-sample reconstruction is biased by ~that mode's
    #    energy. With p = len(Sx) and rank=None, in-sample x_hat_k == x_k
    #    up to floating-point error.
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    p = len(Sx) if rank is None else min(rank, len(Sx))
    Ux = Ux[:, :p]                          # (n, p)  POD basis

    # 4. Split the input modes into state (Uo1) and control (Uo2) parts.
    Uo1 = Uo[:n, :]                         # (n, r)
    Uo2 = Uo[n:, :]                         # (q, r)

    # 5. Reduced operators.
    common = X2 @ Vto.T @ np.diag(1.0 / So)  # (n, r)  - X2 V S^-1
    A = Ux.T @ common @ Uo1.T @ Ux           # (p, p)
    B = Ux.T @ common @ Uo2.T                # (p, q)

    return {"A": A, "B": B, "modes": Ux, "rank": p}


def forecast(model: dict, x0: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
    # Iterate y_{k+1} = A y_k + B u_k in the reduced space, then lift.
    # x0:    (n,)            initial full-state vector
    # U_seq: (q, T)          control inputs to apply
    # returns (n, T+1)       forecast trajectory in full state space
    A, B, modes = model["A"], model["B"], model["modes"]
    y = modes.T @ x0                        # project to reduced coords
    ys = [y]
    for k in range(U_seq.shape[1]):
        y = A @ y + B @ U_seq[:, k]
        ys.append(y)
    Y = np.column_stack(ys)                 # (p, T+1)
    return modes @ Y                        # lift back to full space (n, T+1)
