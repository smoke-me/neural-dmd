"""
Gavish-Donoho optimal hard threshold for singular-value truncation.

Reference:
    M. Gavish & D. L. Donoho, "The Optimal Hard Threshold for Singular
    Values is 4/sqrt(3)", IEEE Trans. Information Theory 60(8), 2014.

Setting: a low-rank signal matrix is observed in additive i.i.d. white
noise. Truncating the SVD at the threshold returned here minimises the
asymptotic mean-square reconstruction error of the signal. The
threshold lives entirely in the spectrum of the observed matrix - no
training-loop information is required.

Two regimes:

  * Known noise sigma:
        tau* = lambda_star(beta) * sqrt(n) * sigma
    where beta = min(m, n) / max(m, n) is the aspect ratio of the
    (m x n) data matrix. Use when an external sigma estimate is
    available (e.g. config.NOISE_SIGMA * trajectory std).

  * Unknown noise (recommended, self-contained from the spectrum):
        tau* = omega(beta) * sigma_hat,    sigma_hat = median(s_i)
    where omega(beta) = lambda_star(beta) / sqrt(mu_beta) and mu_beta
    is the median of the Marchenko-Pastur distribution at aspect beta.

Both regimes return the same data class: threshold tau* + truncation
rank (# of singular values strictly above tau*).

Public surface:
    optimal_threshold(s, *, shape=None, sigma=None) -> GDResult

Implementation notes:
    - lambda_star is closed form (paper Eq. (11)).
    - mu_beta has no closed form; solved by scipy.optimize.brentq on
      the Marchenko-Pastur CDF == 0.5 over its support. The integral is
      evaluated by scipy.integrate.quad.
    - mu_beta is memoised per beta so repeated calls on the same matrix
      shape are O(1) after the first solve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass
class GDResult:
    """Outcome of a Gavish-Donoho threshold computation."""
    threshold:  float    # tau* in singular-value units
    rank:       int      # number of s_i > threshold
    beta:       float    # min(m, n) / max(m, n) aspect ratio used
    sigma_hat:  float    # noise estimate used (median(s) for unknown-sigma)
    omega:      float    # ω(β) coefficient (only for unknown-sigma path)
    lambda_star: float   # λ*(β) coefficient
    sigma_known: bool    # True iff sigma was supplied externally


def lambda_star(beta: float) -> float:
    """Closed-form λ*(β) from Gavish-Donoho Eq. (11)."""
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"lambda_star: beta must be in (0, 1], got {beta}")
    return math.sqrt(
        2.0 * (beta + 1.0)
        + (8.0 * beta) / ((beta + 1.0) + math.sqrt(beta * beta + 14.0 * beta + 1.0))
    )


def _mp_density(x: float, beta: float) -> float:
    """Marchenko-Pastur density f_β(x). Zero outside the support."""
    lo = (1.0 - math.sqrt(beta)) ** 2
    hi = (1.0 + math.sqrt(beta)) ** 2
    if x <= lo or x >= hi or x == 0.0:
        return 0.0
    return math.sqrt((hi - x) * (x - lo)) / (2.0 * math.pi * beta * x)


@lru_cache(maxsize=128)
def marcenko_pastur_median(beta: float) -> float:
    """Median μ_β of the Marchenko-Pastur distribution with aspect β.

    Solves CDF(x; β) = 0.5 over the MP support [(1−√β)^2, (1+√β)^2].
    Memoised per β: first call ~50 ms (one brentq + a few quad calls),
    subsequent calls O(1).
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    if not (0.0 < beta <= 1.0):
        raise ValueError(
            f"marcenko_pastur_median: beta must be in (0, 1], got {beta}")
    lo = (1.0 - math.sqrt(beta)) ** 2
    hi = (1.0 + math.sqrt(beta)) ** 2

    def cdf_minus_half(x: float) -> float:
        v, _ = quad(_mp_density, lo, x, args=(beta,), limit=200)
        return v - 0.5

    # CDF is monotone non-decreasing on [lo, hi]; brentq is bullet-proof.
    return float(brentq(cdf_minus_half, lo, hi, xtol=1e-12))


def omega(beta: float) -> float:
    """ω(β) = λ*(β) / sqrt(μ_β). The unknown-sigma coefficient."""
    return lambda_star(beta) / math.sqrt(marcenko_pastur_median(beta))


def optimal_threshold(s, *,
                      shape: tuple[int, int] | None = None,
                      sigma: float | None = None) -> GDResult:
    """Compute the Gavish-Donoho optimal hard threshold for a singular
    spectrum `s`.

    Parameters
    ----------
    s : 1-D array of singular values (in descending order, as returned
        by numpy.linalg.svd).
    shape : (m, n) of the original matrix that produced `s`. Used to
        derive β = min(m, n) / max(m, n). If None, β is inferred as
        1.0 (square-matrix assumption); supply this when m != n to get
        the correct anisotropy correction.
    sigma : optional known noise std. When provided the known-σ formula
        is used (tau* = lambda_star * sqrt(max(m,n)) * sigma). When
        None, the median-based unknown-σ estimator is used.

    Returns a GDResult with the threshold, the integer truncation rank,
    and diagnostics suitable for logging.
    """
    s = np.asarray(s, dtype=float).ravel()
    if s.size == 0:
        raise ValueError("optimal_threshold: empty singular-value array")

    if shape is None:
        beta = 1.0
        n_big = s.size
    else:
        m, n = int(shape[0]), int(shape[1])
        if m <= 0 or n <= 0:
            raise ValueError(f"optimal_threshold: invalid shape={shape!r}")
        beta = min(m, n) / max(m, n)
        n_big = max(m, n)

    ls = lambda_star(beta)

    if sigma is not None:
        sigma_hat = float(sigma)
        threshold = ls * math.sqrt(n_big) * sigma_hat
        om = float("nan")
        sigma_known = True
    else:
        sigma_hat = float(np.median(s))
        om = omega(beta)
        threshold = om * sigma_hat
        sigma_known = False

    rank = int(np.sum(s > threshold))
    return GDResult(threshold=float(threshold),
                    rank=int(rank),
                    beta=float(beta),
                    sigma_hat=float(sigma_hat),
                    omega=float(om),
                    lambda_star=float(ls),
                    sigma_known=bool(sigma_known))
