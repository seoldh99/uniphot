"""Noise: correlated Gaussian sky (separable 3-tap kernel) and source Poisson noise.

Sky: white N(0, 1) noise convolved with K = k (x) k / sum(k^2), k = [a, 1, a], where a
solves 2a / (1 + 2a^2) = rho1.  Then the per-pixel sigma is exactly ``sigma`` and the
correlation at one pixel along a row or a column is exactly rho1.  rho1 = 0 gives white noise.
"""
from __future__ import annotations

import numpy as np

RHO1 = 0.15


def corr_kernel_1d(rho1: float = RHO1) -> np.ndarray:
    """1-D taps k = [a, 1, a] / sqrt(1 + 2a^2) (unit sum of squares) for lag-1 correlation rho1."""
    if not 0.0 <= rho1 < 1.0 / np.sqrt(2.0):
        raise ValueError(f"rho1 must be in [0, 0.707) for a 3-tap kernel, not {rho1!r}")
    a = 0.0 if rho1 == 0.0 else (1.0 - np.sqrt(1.0 - 2.0 * rho1 ** 2)) / (2.0 * rho1)
    return np.array([a, 1.0, a]) / np.sqrt(1.0 + 2.0 * a * a)


def autocorrelation(lag_x: int, lag_y: int, rho1: float = RHO1) -> float:
    """Correlation between two sky pixels separated by (lag_x, lag_y) pixels."""
    k = corr_kernel_1d(rho1)
    r = {0: float(k @ k), 1: float(k[0] * k[1] + k[1] * k[2]), 2: float(k[0] * k[2])}
    return r.get(abs(int(lag_x)), 0.0) * r.get(abs(int(lag_y)), 0.0)


def sky_noise(shape, rng: np.random.Generator, sigma: float = 1.0,
              rho1: float = RHO1) -> np.ndarray:
    """Stationary correlated Gaussian sky noise of the given shape (also at the edges).

    shape: (..., H, W); the last two axes are image axes, leading axes index realisations.
    """
    shape = tuple(int(s) for s in shape)
    if rho1 == 0.0:
        return sigma * rng.standard_normal(shape)
    k = corr_kernel_1d(rho1)
    h, w = shape[-2:]
    white = rng.standard_normal(shape[:-2] + (h + 2, w + 2))
    rows = k[0] * white[..., :, :w] + k[1] * white[..., :, 1:w + 1] + k[2] * white[..., :, 2:w + 2]
    sky = k[0] * rows[..., :h, :] + k[1] * rows[..., 1:h + 1, :] + k[2] * rows[..., 2:h + 2, :]
    return sigma * sky


def poisson_counts(model: np.ndarray, gain: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson(gain * model) / gain; model must be >= 0 (image units)."""
    model = np.asarray(model, dtype=float)
    if np.any(model < 0) or not np.all(np.isfinite(model)):
        raise ValueError("the source model must be finite and >= 0 for Poisson noise")
    return rng.poisson(gain * model) / gain


def realize(model: np.ndarray, rng: np.random.Generator, n: int | None = None,
            sigma: float = 1.0, gain: float | None = 50.0, rho1: float = RHO1) -> np.ndarray:
    """Noisy realisation(s) of a noiseless model: Poisson(g model) / g + sky.

    n None: one image of model.shape; n given: a stack (n, *model.shape) of independent
    realisations.  gain None: no source Poisson noise.
    """
    model = np.asarray(model, dtype=float)
    shape = model.shape if n is None else (int(n), *model.shape)
    base = np.broadcast_to(model, shape)
    out = poisson_counts(base, gain, rng) if gain is not None else np.array(base, dtype=float)
    if sigma > 0.0:
        out = out + sky_noise(shape, rng, sigma=sigma, rho1=rho1)
    return out


def linear_estimator_variance(weights: np.ndarray, sigma: float = 1.0,
                              rho1: float = RHO1) -> float:
    """Variance of sum(weights * sky) for this sky model (the sky part only)."""
    w = np.asarray(weights, dtype=float)
    h, wd = w.shape
    tot = 0.0
    for lx in range(-2, 3):
        for ly in range(-2, 3):
            c = autocorrelation(lx, ly, rho1)
            if c == 0.0:
                continue
            a = w[max(0, ly):h + min(0, ly), max(0, lx):wd + min(0, lx)]
            b = w[max(0, -ly):h + min(0, -ly), max(0, -lx):wd + min(0, -lx)]
            tot += c * float((a * b).sum())
    return sigma ** 2 * tot
