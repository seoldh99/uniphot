"""Synthetic XP magnitudes and truth zero-point surfaces.

m_XP = ZP(xh, yh) - 2.5 log10(F_box) + N(0, sigma_mag^2) [+ outlier_shift for outliers].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ZPSurface:
    """ZP(xh, yh) = sum_{(p, q)} c[(p, q)] xh^p yh^q (mag)."""
    shape: tuple
    coeffs: dict

    def __call__(self, x, y):
        """ZP (mag) at image positions (x, y)."""
        h, w = self.shape
        xh = 2.0 * np.asarray(x, dtype=float) / (w - 1) - 1.0
        yh = 2.0 * np.asarray(y, dtype=float) / (h - 1) - 1.0
        return sum(c * xh ** p * yh ** q for (p, q), c in self.coeffs.items()) + 0.0 * xh


def zp_z1(shape, zp0: float = 25.0) -> ZPSurface:
    """Design Z1 surface: zp0 + 0.05 xh (0.05 mag at the field edge) + 0.02 yh^2."""
    return ZPSurface(tuple(shape), {(0, 0): float(zp0), (1, 0): 0.05, (0, 2): 0.02})


@dataclass(frozen=True)
class XP:
    """Synthetic XP magnitudes and the outlier mask."""
    mag: np.ndarray
    outlier: np.ndarray


def synthetic_xp(flux, x, y, zp: ZPSurface, rng: np.random.Generator, sigma_mag: float = 0.01,
                 outlier_fraction: float = 0.0, outlier_shift: float = 0.3) -> XP:
    """XP magnitudes of sources with truth box flux ``flux`` at (x, y); round(outlier_fraction n)
    sources get +outlier_shift mag."""
    flux = np.atleast_1d(np.asarray(flux, dtype=float))
    n = len(flux)
    mag = zp(x, y) - 2.5 * np.log10(flux)
    if sigma_mag > 0.0:
        mag = mag + rng.normal(0.0, sigma_mag, size=n)
    outlier = np.zeros(n, dtype=bool)
    n_out = int(round(outlier_fraction * n))
    if n_out:
        idx = rng.choice(n, size=n_out, replace=False)
        mag[idx] += outlier_shift
        outlier[idx] = True
    return XP(mag, outlier)
