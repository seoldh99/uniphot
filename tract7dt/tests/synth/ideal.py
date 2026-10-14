"""Ideal implementations: truth PSF at the truth position, zero background.

These are the reference fits the false-failure device runs (design 6: "the test code measures the
false-failure rate once with the ideal implementation (a fit given the true PSF and the true transform)
and records it").
"""
from __future__ import annotations

import numpy as np

from .noise import RHO1, linear_estimator_variance
from .psf import BOX, HALF, PSFField


def psf_flux(data: np.ndarray, stamp: np.ndarray, ix0: int, iy0: int) -> np.ndarray:
    """Sky-weighted PSF flux sum(phi d) / sum(phi^2) with the truth stamp at its box.

    data: (H, W) image or (n, H, W) stack; the box must lie inside the image.
    """
    data = np.asarray(data, dtype=float)
    h, w = data.shape[-2:]
    if not (HALF <= ix0 <= w - 1 - HALF and HALF <= iy0 <= h - 1 - HALF):
        raise ValueError("the box must lie inside the image")
    cut = data[..., iy0 - HALF:iy0 + HALF + 1, ix0 - HALF:ix0 + HALF + 1]
    return (cut * stamp).sum(axis=(-2, -1)) / (stamp ** 2).sum()


def poisson_factor(stamp: np.ndarray) -> float:
    """sum(phi^3) / sum(phi^2)^2: Poisson variance of psf_flux in units of F / g."""
    return float((stamp ** 3).sum() / (stamp ** 2).sum() ** 2)


def flux_variance(stamp: np.ndarray, flux: float, sigma: float = 1.0, gain: float | None = 50.0,
                  rho1: float = RHO1) -> float:
    """Expected variance of psf_flux: sky part (with correlation) + F sum(phi^3) / (g sum(phi^2)^2)."""
    s2 = (stamp ** 2).sum()
    sky = linear_estimator_variance(stamp / s2, sigma=sigma, rho1=rho1)
    return float(sky if gain is None else sky + flux * (stamp ** 3).sum() / (gain * s2 ** 2))


def sky_limited_flux(snr, stamp: np.ndarray, sigma: float = 1.0, rho1: float = RHO1):
    """Flux whose ideal PSF-fit S/N from the sky alone is ``snr``."""
    sky = linear_estimator_variance(stamp / (stamp ** 2).sum(), sigma=sigma, rho1=rho1)
    return np.asarray(snr, dtype=float) * np.sqrt(sky)


def fixed_offset_flux_ratio(psf: PSFField, x: float, y: float, dx: float, dy: float) -> float:
    """Ideal PSF flux / truth flux when the model sits at (x + dx, y + dy) but the star is at (x, y).

    Noise-free; the model box is centred on the model position.  1 - ratio is the flux loss of a
    fit held at a wrong position (design A4: 0.5 px, Moffat beta 3, FWHM 4.3 px).
    """
    s = psf.stamp(x, y)
    m = psf.stamp(x + dx, y + dy)
    ox, oy = min(s.ix0, m.ix0) - HALF, min(s.iy0, m.iy0) - HALF
    w = max(s.ix0, m.ix0) + HALF + 1 - ox
    h = max(s.iy0, m.iy0) + HALF + 1 - oy
    a = np.zeros((h, w))
    b = np.zeros((h, w))
    a[s.iy0 - HALF - oy:s.iy0 - HALF - oy + BOX, s.ix0 - HALF - ox:s.ix0 - HALF - ox + BOX] = s.data
    b[m.iy0 - HALF - oy:m.iy0 - HALF - oy + BOX, m.ix0 - HALF - ox:m.ix0 - HALF - ox + BOX] = m.data
    return float((a * b).sum() / (b * b).sum())
