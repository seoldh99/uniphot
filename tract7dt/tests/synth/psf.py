"""Truth PSF: pixel-integrated Moffat / Gaussian with the design FWHM map and ring corner.

Every profile below has unit integral over the whole plane.  ``pixel_integrate`` integrates
a profile over each pixel of a square box with Gauss-Legendre quadrature (``order`` nodes
per axis per pixel); ``PSFField.stamp`` then divides by the box sum (box flux convention).
Order 8 (default) matches exact erf pixel integrals of a Gaussian to < 1e-16 for FWHM >= 2 px
(order 6 misses by 1.3e-13 at FWHM 2.5; impl_0.3.md) and the design PSFs agree with order 16 to
< 1e-10 of the peak (test_synth_generator.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import special

BOX = 55
HALF = BOX // 2  # 27
_FWHM_GAUSS = 2.0 * np.sqrt(2.0 * np.log(2.0))


def moffat_alpha(fwhm, beta):
    """Moffat core radius alpha for a given FWHM: FWHM = 2 alpha sqrt(2^(1/beta) - 1)."""
    return np.asarray(fwhm, dtype=float) / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def _int_power(t, n: int):
    """t^n by repeated products (numpy's t ** 3 takes the slow general-power path)."""
    out = t
    for _ in range(n - 1):
        out = out * t
    return out


def _moffat(u, v, fwhm, beta):
    a2 = moffat_alpha(fwhm, beta) ** 2
    t = 1.0 / (1.0 + (u * u + v * v) / a2)
    k2 = 2.0 * beta
    if k2 == int(k2) and 2 <= k2 <= 16:      # integer or half-integer beta (3, 2.5): products
        n = int(beta)
        p = _int_power(t, n) if beta == n else _int_power(t, n) * np.sqrt(t)
    else:
        p = t ** beta
    return (beta - 1.0) / (np.pi * a2) * p


def _gaussian(u, v, fwhm):
    s2 = (fwhm / _FWHM_GAUSS) ** 2
    return np.exp(-(u * u + v * v) / (2.0 * s2)) / (2.0 * np.pi * s2)


def _ring(u, v, radius, sigma):
    norm = 2.0 * np.pi * (sigma ** 2 * np.exp(-radius ** 2 / (2.0 * sigma ** 2))
                          + np.sqrt(2.0 * np.pi) * radius * sigma * special.ndtr(radius / sigma))
    return np.exp(-(np.hypot(u, v) - radius) ** 2 / (2.0 * sigma ** 2)) / norm


def box_center(x):
    """Pixel index of the box centre for position x (numpy rint, half to even)."""
    return np.rint(np.asarray(x, dtype=float)).astype(np.int64)


_GL_CACHE: dict = {}


def _gl(order: int):
    """Gauss-Legendre nodes and weights on [-0.5, 0.5]."""
    if order not in _GL_CACHE:
        t, w = np.polynomial.legendre.leggauss(order)
        _GL_CACHE[order] = (0.5 * t, 0.5 * w)
    return _GL_CACHE[order]


def pixel_integrate(profile: Callable, dx: float, dy: float, size: int = BOX,
                    order: int = 8) -> np.ndarray:
    """Integrate ``profile(u, v)`` over each pixel of a size x size box.

    (u, v) are offsets from the source; the source sits at (dx, dy) from the centre pixel
    of the box.  Returns array[j, i] = integral over pixel (i, j) of the box.
    """
    t, w = _gl(order)
    c = np.arange(size, dtype=float) - size // 2
    u = (c[:, None] + t[None, :] - dx).ravel()          # (size * order,)
    v = (c[:, None] + t[None, :] - dy).ravel()
    f = np.asarray(profile(u[None, :], v[:, None]), dtype=float)
    f = f.reshape(size, order, size, order)             # [j, b, i, a]
    return np.tensordot(np.tensordot(f, w, axes=([3], [0])), w, axes=([1], [0]))


@dataclass(frozen=True)
class Stamp:
    """A box-normalised truth PSF stamp.

    data: (55, 55) array, sums to 1.  (ix0, iy0): image pixel of the box centre.
    box_fraction: share of the unit-integral continuous PSF that falls inside the box.
    """
    data: np.ndarray
    ix0: int
    iy0: int
    box_fraction: float


@dataclass(frozen=True)
class PSFField:
    """Spatially varying truth PSF of one synthetic image.

    shape: (H, W) of the image.  profile: "moffat" or "gaussian".  fwhm: None for the design
    map 3.7 + (xh - yh)^2 px, a number for a constant FWHM.  ring_corner: blend in the ring
    in the corner xh > 0.6 and yh > 0.6.  order: Gauss-Legendre nodes per axis per pixel.
    """
    shape: tuple
    profile: str = "moffat"
    beta: float = 3.0
    fwhm: float | None = None
    ring_corner: bool = False
    ring_radius: float = 10.0
    ring_sigma: float = 1.5
    order: int = 8

    def __post_init__(self):
        if self.profile not in ("moffat", "gaussian"):
            raise ValueError(f"profile must be 'moffat' or 'gaussian', not {self.profile!r}")
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError(f"shape must be (H, W) with H, W >= 2, not {self.shape!r}")

    def xyhat(self, x, y):
        """Field coordinates (xh, yh) in [-1, 1] of image position (x, y)."""
        h, w = self.shape
        return (2.0 * np.asarray(x, dtype=float) / (w - 1) - 1.0,
                2.0 * np.asarray(y, dtype=float) / (h - 1) - 1.0)

    def fwhm_at(self, x, y, fwhm_scale=1.0):
        """FWHM (px) of the Moffat/Gaussian component at (x, y), times ``fwhm_scale``."""
        xh, yh = self.xyhat(x, y)
        base = 3.7 + (xh - yh) ** 2 if self.fwhm is None else np.full_like(xh, float(self.fwhm))
        return base * fwhm_scale

    def ring_weight_at(self, x, y):
        """Weight w of the ring component at (x, y) (0 when ring_corner is False)."""
        xh, yh = self.xyhat(x, y)
        if not self.ring_corner:
            return np.zeros_like(xh)
        return 0.6 * np.clip((np.minimum(xh, yh) - 0.6) / 0.4, 0.0, 1.0)

    def _profile(self, x, y, fwhm_scale=1.0):
        fwhm = float(self.fwhm_at(x, y, fwhm_scale))
        w = float(self.ring_weight_at(x, y))
        if self.profile == "moffat":
            core = lambda u, v: _moffat(u, v, fwhm, self.beta)
        else:
            core = lambda u, v: _gaussian(u, v, fwhm)
        if w == 0.0:
            return core
        r0, s0 = self.ring_radius, self.ring_sigma
        return lambda u, v: (1.0 - w) * core(u, v) + w * _ring(u, v, r0, s0)

    def continuous(self, x, y, u, v, fwhm_scale=1.0):
        """Unit-integral continuous PSF of a source at (x, y), at offsets (u, v) from it."""
        return self._profile(x, y, fwhm_scale)(np.asarray(u, dtype=float), np.asarray(v, dtype=float))

    def stamp(self, x, y, fwhm_scale=1.0) -> Stamp:
        """Box-normalised pixel-integrated truth PSF of a source at image position (x, y)."""
        ix0, iy0 = int(box_center(x)), int(box_center(y))
        raw = pixel_integrate(self._profile(x, y, fwhm_scale), float(x) - ix0, float(y) - iy0,
                              BOX, self.order)
        frac = float(raw.sum())
        return Stamp(raw / frac, ix0, iy0, frac)
