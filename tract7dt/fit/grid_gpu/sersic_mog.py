"""JAX runtime for the dense Sersic mixture-of-Gaussians table.

Loads ``tract7dt/data/sersic_mog_table.npz`` (built by
:mod:`tract7dt.fit.grid_gpu.sersic_mog_gen` from The Tractor's
``SersicMixture``; the Tractor is needed only to rebuild the table) and provides differentiable ``n -> (amps, vars, core)``
lookup plus the FFT-based galaxy renderer used for extended sources.

Design notes:

- The table is piecewise per vendor spline segment with DUPLICATED knots at
  segment boundaries; ``jnp.searchsorted(..., side='right') - 1`` lands each
  query inside one segment so interpolation never spans a boundary pair.
  Linear interpolation on the dn=0.005 grid reproduces Tractor to <=1.2e-3
  relative on the radial profile (verified offline); gradients d/dn are exact
  for the interpolant.
- Total flux is per-family: the Tractor rows (n < 0.8) satisfy
  sum(amps) + core == 1 exactly (to ~9e-16); the Shajib rows (n >= 0.8) sum
  to 0.9534-1.0047 BY DESIGN -- the deficit is far-wing flux beyond the
  widest sigma, deliberately NOT renormalized (in-window SB is what LSQ
  fitting sees; the fitted amplitude then estimates the untruncated total
  directly). Renderers built on this module must never renormalize by a box
  sum -- that is precisely the flux-truncation defect this module exists to
  fix.
- The delta core (n > 1) renders as ``core * PSF`` and is added AFTER the
  Gaussian part is convolved.

TODO (island-arch P3 finding, 2026-07-05): ``get_table_jax()`` caches its
device arrays lazily on first call -- if that first call happens INSIDE a
jit trace (e.g. an island-solver fft-render kernel), the cache captures a
TRACER and every later trace dies with UnexpectedTracerError. The full
pipeline is safe only by accident (S1's eager render runs first);
island_solver.py works around it with an eager warmup call. Proper fix:
populate the cache with concrete arrays at module import (or first
NON-traced call). Do not restructure until the island work lands.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

K_MAX = 15

_TABLE_PATH = Path(__file__).resolve().parents[2] / "data" / "sersic_mog_table.npz"
_np_cache = None


def load_table_np():
    """Load the table as plain numpy arrays (cached)."""
    global _np_cache
    if _np_cache is None:
        z = np.load(_TABLE_PATH)
        _np_cache = {k: z[k] for k in z.files}
    return _np_cache


class SersicTableJax:
    """Device-resident table. Construct once per process/worker."""

    def __init__(self):
        import jax.numpy as jnp
        t = load_table_np()
        self.n_knots = jnp.asarray(t["n_knots"])
        self.amps = jnp.asarray(t["amps"])
        self.log_vars = jnp.asarray(t["log_vars"])
        self.core_amp = jnp.asarray(t["core_amp"])
        self.n_min = float(t["n_min"])
        self.n_max = float(t["n_max"])


_jax_table = None


def get_table_jax() -> SersicTableJax:
    global _jax_table
    if _jax_table is None:
        _jax_table = SersicTableJax()
    return _jax_table


def mog_params_at(table: SersicTableJax, n):
    """Differentiable ``n -> (amps[K_MAX], vars[K_MAX], core)`` lookup.

    ``n`` is clipped to the table's hard bounds (matching Tractor's clamp).
    """
    import jax.numpy as jnp
    n = jnp.clip(n, table.n_min, table.n_max)
    # 'right' - 1: inside a segment; at a duplicated boundary knot pair the
    # query n == knot value lands on the RIGHT segment's first row, whose
    # mixture equals the left segment's last row (continuity by construction).
    i = jnp.searchsorted(table.n_knots, n, side="right") - 1
    i = jnp.clip(i, 0, table.n_knots.shape[0] - 2)
    n0 = table.n_knots[i]
    n1 = table.n_knots[i + 1]
    # duplicated pair (n1 == n0) can only be hit when n == n0 exactly -> t=0.
    t = jnp.where(n1 > n0, (n - n0) / jnp.where(n1 > n0, n1 - n0, 1.0), 0.0)
    amps = (1.0 - t) * table.amps[i] + t * table.amps[i + 1]
    log_vars = (1.0 - t) * table.log_vars[i] + t * table.log_vars[i + 1]
    core = (1.0 - t) * table.core_amp[i] + t * table.core_amp[i + 1]
    return amps, jnp.exp(log_vars), core


# ---------------------------------------------------------------------------
# Ellipse machinery (shared convention with source_model.render_exp_profile_
# stamp_jax: EllipseESoft (logre, ee1, ee2) -> per-component pixel covariance)
# ---------------------------------------------------------------------------

def _ellipse_cov_terms(logre, ee1, ee2):
    """Return (cxx, cyy, cxy) unit-variance covariance factors such that a
    mixture component with variance ``v`` (units re**2) has pixel covariance
    ``[[v*cxx, v*cxy], [v*cxy, v*cyy]]`` with x the fast (W) stamp axis.

    This MUST mirror the (a, b, c, d) affine in
    ``source_model.render_exp_profile_stamp_jax`` exactly (c00 pairs with the
    x axis there): a = re*ct/inv_ba, b = re*st, c = -re*st/inv_ba, d = re*ct;
    c00 = v*(a*a + b*b), c01 = v*(a*c + b*d), c11 = v*(c*c + d*d).
    """
    import jax.numpy as jnp
    re = jnp.clip(jnp.exp(logre), 0.1, 1000.0)
    # ROUND-SOURCE (ELL=0) JVP GUARD (2026-07-06): the polar
    # decomposition below (hypot + arctan2) has a 0/0 forward-mode
    # derivative at EXACTLY ee1 = ee2 = 0. A catalog row with ELL=0
    # seeds ee1 = ee2 = 0 (ellipse_soft_from_re_ab_phi_np), so one
    # perfectly-round extended source NaN-poisons the joint-varpro
    # Jacobian of its whole grid and silently freezes that grid's
    # positions/shapes at the seeds (root-caused on the island-P3
    # T08285 injections; same singularity class as the STAR-row guard
    # in joint_lsqr, but extended rows cannot be masked upstream).
    # Nudge exact zeros to a dynamically invisible ellipticity (axis
    # ratio 0.9999): the where() selects the constant branch at the
    # singular point, so d/d(ee1) is 0 there (instead of NaN) and
    # d/d(ee2) stays finite. The primal changes ONLY for exactly-zero
    # rows; every other input is bit-identical. Star-only catalogs
    # never reach this code path (direct backend).
    ee1 = jnp.where((ee1 == 0.0) & (ee2 == 0.0),
                    jnp.asarray(1e-4, dtype=jnp.result_type(ee1)), ee1)
    ee = jnp.hypot(ee1, ee2)
    e = 1.0 - jnp.exp(-ee)
    inv_ba = jnp.minimum((1.0 + e) / jnp.maximum(1.0 - e, 1e-4), 1000.0)
    theta = 0.5 * jnp.arctan2(ee2, ee1)
    ct = jnp.cos(theta)
    st = jnp.sin(theta)
    a = re * ct / inv_ba
    b = re * st
    c = -re * st / inv_ba
    d = re * ct
    cxx = a * a + b * b
    cyy = c * c + d * d
    cxy = a * c + b * d
    return cxx, cyy, cxy


def render_mog_stamp_fft(psf_stamp, amps, vars_re2, core, logre, ee1, ee2,
                         pad: int):
    """Render a unit-total-flux MoG galaxy convolved with a pixelized PSF.

    Mechanism (Tractor galaxy.py FFT path, Lang 2020): the galaxy is never
    rasterized -- its Fourier transform is the closed-form zero-mean Gaussian
    sum evaluated on the PSF stamp's padded frequency grid; multiply by
    rfft2(PSF), irfft2 back, crop to the stamp, add ``core * PSF``.

    Pixel integration is EXACT because the PSFEx-style stamp is already
    pixel-integrated. Output integrates to (1 - wings-outside-stamp) --
    callers must NOT renormalize.

    Parameters
    ----------
    psf_stamp : (H, W) array, unit-sum pixelized PSF at the source position.
    amps, vars_re2 : (K_MAX,) mixture amplitudes / variances (re**2 units).
    core : scalar delta-core amplitude.
    logre, ee1, ee2 : EllipseESoft shape.
    pad : static padded FFT size (power of two >= 2 * ceil(max(H, W) / 2)).
    """
    import jax.numpy as jnp
    H, W = psf_stamp.shape
    cxx, cyy, cxy = _ellipse_cov_terms(logre, ee1, ee2)

    psf_pad = jnp.zeros((pad, pad), dtype=psf_stamp.dtype)
    psf_pad = psf_pad.at[:H, :W].set(psf_stamp)
    P = jnp.fft.rfft2(psf_pad)

    v = jnp.fft.rfftfreq(pad)          # (pad//2+1,)
    w = jnp.fft.fftfreq(pad)           # (pad,)
    vv = v[None, :]
    ww = w[:, None]
    # zero-mean Gaussian FT: exp(-2 pi^2 (Cxx v^2 + Cyy w^2 + 2 Cxy v w))
    quad = (cxx * vv * vv + cyy * ww * ww + 2.0 * cxy * vv * ww)
    expo = -2.0 * (jnp.pi ** 2) * quad
    # sum over components: amps_k * exp(vars_k * expo); zero-amp pads no-op.
    F = jnp.sum(amps[:, None, None] * jnp.exp(vars_re2[:, None, None] * expo),
                axis=0)
    conv = jnp.fft.irfft2(P * F, s=(pad, pad))
    stamp = conv[:H, :W] + core * psf_stamp
    return stamp


def fft_pad_for(stamp_hw: int) -> int:
    """Smallest power of two >= stamp + stamp//2 (half-stamp wraparound margin).

    For the default 55-px stamp this returns 128 (55 + 27 = 82 -> next pow2),
    NOT Tractor's 64 -- the extra margin halves the periodic-convolution
    wraparound error vs a 64 pad; large stamp tiers (111/221) land on 256/512.
    """
    n = 1
    while n < stamp_hw + stamp_hw // 2:
        n <<= 1
    return n
