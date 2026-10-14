"""Synthetic truth generator for the uniphot correctness tests (M2 unit 0.3).

Design: the development design note ``design_v2.md`` (not distributed), section 6 (common conventions);
plan: ``impl_plan.md`` 1.2 (unit 0.3).

This package does not import the engine (``tract7dt``) or JAX.  The truth it makes is
defined by analytic functions that are independent of the engine's models (design 6:
stars planted with the engine's own PSF are only a wiring check).
``test_synth_isolation.py`` checks this statically and in a fresh process.
Tests import it as ``synth`` (pytest puts ``tract7dt/tests`` on ``sys.path``).

Conventions (positions in pixels, fluxes in image units)
- Pixel (i, j) is the square x in [i - 0.5, i + 0.5], y in [j - 0.5, j + 0.5]; arrays are
  indexed ``data[j, i]``; coordinates are 0-based, so pixel centres sit on integers.
- Field coordinates: xh = 2 x / (W - 1) - 1 and yh = 2 y / (H - 1) - 1, both in [-1, 1].
- Box: the 55 x 55 pixel square centred on pixel (rint(x), rint(y)) (numpy rounding,
  half to even).  Truth flux = box flux: every source's truth PSF is normalised to sum 1
  inside its box and is not rendered outside it.
- Truth PSF: pixel-integrated Moffat (beta 3 by default) with FWHM(xh, yh) = 3.7 + (xh - yh)^2
  px, optionally blended with a ring (radius 10 px, Gaussian thickness sigma 1.5 px) in the
  corner xh > 0.6 and yh > 0.6 with weight w = 0.6 clip((min(xh, yh) - 0.6) / 0.4, 0, 1).
- Noise: data = Poisson(g model) / g + sky with g = 50 e-/unit; the sky is Gaussian with
  per-pixel sigma 1 and a separable 3-tap correlation kernel with rho(1 px) = 0.15.
"""
from __future__ import annotations

from .astrometry import OffsetField, Seeds, gaia_seeds, offset_a1
from .ideal import (fixed_offset_flux_ratio, flux_variance, poisson_factor, psf_flux,
                    sky_limited_flux)
from .noise import (RHO1, autocorrelation, corr_kernel_1d, linear_estimator_variance,
                    poisson_counts, realize, sky_noise)
from .psf import BOX, HALF, PSFField, Stamp, box_center, moffat_alpha, pixel_integrate
from .scene import (DENSITY_CROWDED, SPARSE_N_RANGE, SynthImage, crowded_positions, draw_positions,
                    draw_snr, make_image, render, star_count)
from .xp import XP, ZPSurface, synthetic_xp, zp_z1

__all__ = [
    "BOX", "HALF", "PSFField", "Stamp", "box_center", "moffat_alpha", "pixel_integrate",
    "RHO1", "autocorrelation", "corr_kernel_1d", "linear_estimator_variance",
    "poisson_counts", "realize", "sky_noise",
    "DENSITY_CROWDED", "SPARSE_N_RANGE", "SynthImage", "crowded_positions", "draw_positions", "draw_snr",
    "make_image", "render", "star_count",
    "OffsetField", "Seeds", "gaia_seeds", "offset_a1",
    "XP", "ZPSurface", "synthetic_xp", "zp_z1",
    "fixed_offset_flux_ratio", "flux_variance", "poisson_factor", "psf_flux",
    "sky_limited_flux",
]
