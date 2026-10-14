"""Source-type helpers for the grid GPU fitter.

The CPU path builds Tractor ``PointSource`` and ``ExpGalaxy`` objects from
the input catalog.  The grid GPU path keeps that state as arrays so JAX can
render source stamps and update parameters in bulk.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from ...utils import case_insensitive_col_map
from .psf_polynomial import evaluate_psf_at, evaluate_psf_at_batch
from ..optimize.gpu.kernels.lanczos import lanczos_shift_image

logger = logging.getLogger("tract7dt.grid_gpu.source_model")


SOURCE_STAR = 0
SOURCE_EXP = 1
SOURCE_DEV = 2
SOURCE_SERSIC = 3

# Default Sersic indices per catalog TYPE. EXP renders from EXP_AMP/EXP_VAR
# below (NOT the table row at n=1 -- different provenance, ~0.16% apart);
# DEV/SERSIC render from the sersic_mog table at these n seeds.
DEFAULT_N_DEV = 4.0
DEFAULT_N_SERSIC = 2.0
SERSIC_N_MIN = 0.29
SERSIC_N_MAX = 6.3


def sersic_n_to_u(n):
    """Map n in (SERSIC_N_MIN, SERSIC_N_MAX) to an UNCONSTRAINED internal
    parameter via the logit bijection (AstroPhot-style smooth bounds: the
    solver never hits a hard wall, gradients stay finite, and the mixture
    table's clip never activates during optimization). numpy or jax input."""
    import numpy as _np
    xp = _np if isinstance(n, (float, int, _np.ndarray)) else __import__("jax.numpy", fromlist=["jnp"])
    t = (xp.clip(n, SERSIC_N_MIN + 1e-4, SERSIC_N_MAX - 1e-4) - SERSIC_N_MIN) / (
        SERSIC_N_MAX - SERSIC_N_MIN)
    return xp.log(t / (1.0 - t))


def sersic_u_to_n(u):
    """Inverse of :func:`sersic_n_to_u` (sigmoid back onto the open range)."""
    import numpy as _np
    xp = _np if isinstance(u, (float, int, _np.ndarray)) else __import__("jax.numpy", fromlist=["jnp"])
    return SERSIC_N_MIN + (SERSIC_N_MAX - SERSIC_N_MIN) / (1.0 + xp.exp(-u))

# Tractor's exponential-profile MoG constants, copied from Tractor's
# tractor.mixture_profiles.  Keeping them here means that neither the package
# nor its JAX worker traces import Tractor.
EXP_AMP = np.array([
    2.34853813e-03,
    3.07995260e-02,
    2.23364214e-01,
    1.17949102e00,
    4.33873750e00,
    5.99820770e00,
], dtype=np.float32)
EXP_AMP = (EXP_AMP / np.sum(EXP_AMP)).astype(np.float32)
EXP_VAR = np.array([
    1.20078965e-03,
    8.84526493e-03,
    3.91463084e-02,
    1.39976817e-01,
    4.60962500e-01,
    1.50159566e00,
], dtype=np.float32)


def is_supported_source_type(type_val: Any) -> bool:
    """Return true for the source types the grid GPU fitter handles."""
    if type_val is None:
        return True
    s = str(type_val).strip().upper()
    return s in ("", "STAR", "POINTSOURCE", "POINT", "PSF", "EXP",
                 "DEV", "SERSIC", "SER")


def _is_exp_type(type_val: Any) -> bool:
    return str(type_val).strip().upper() == "EXP"


def _is_dev_type(type_val: Any) -> bool:
    return str(type_val).strip().upper() == "DEV"


def _is_sersic_type(type_val: Any) -> bool:
    return str(type_val).strip().upper() in ("SERSIC", "SER")


def _cpu_phi_from_theta(theta_deg: np.ndarray) -> np.ndarray:
    theta_clean = np.nan_to_num(theta_deg, nan=0.0)
    return -np.where(theta_clean >= 0, 90.0 - theta_clean, -90.0 - theta_clean)


def ellipse_soft_from_re_ab_phi_np(
    re_pix: np.ndarray,
    ab: np.ndarray,
    phi_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``EllipseESoft.fromRAbPhi``."""
    re = np.clip(np.asarray(re_pix, dtype=np.float64), 0.1, 1000.0)
    ba = np.clip(np.asarray(ab, dtype=np.float64), 0.1, 1.0)
    inv_ba = 1.0 / ba
    e = (inv_ba - 1.0) / (inv_ba + 1.0)
    e = np.clip(e, 0.0, 0.999999)
    ee = -np.log(np.maximum(1.0 - e, 1e-12))
    angle = np.deg2rad(2.0 * (-np.asarray(phi_deg, dtype=np.float64)))
    logre = np.log(re)
    ee1 = ee * np.cos(angle)
    ee2 = ee * np.sin(angle)
    return logre.astype(np.float32), ee1.astype(np.float32), ee2.astype(np.float32)


def ellipse_soft_to_re_ab_phi_np(
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert softened ellipse params back to ``Re``, axis ratio, ``phi``."""
    logre = np.asarray(logre, dtype=np.float64)
    ee1 = np.asarray(ee1, dtype=np.float64)
    ee2 = np.asarray(ee2, dtype=np.float64)
    re = np.exp(logre)
    ee = np.hypot(ee1, ee2)
    e = 1.0 - np.exp(-ee)
    ab = (1.0 - e) / np.maximum(1.0 + e, 1e-12)
    phi = -np.rad2deg(np.arctan2(ee2, ee1) / 2.0)
    phi = np.mod(phi, 180.0)
    return re.astype(np.float32), ab.astype(np.float32), phi.astype(np.float32)


def catalog_source_params(
    catalog: pd.DataFrame,
    *,
    re_fallback_pix: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build source type, initial morphology, and Sersic-n arrays from a catalog."""
    col_map = case_insensitive_col_map(catalog.columns)
    n = len(catalog)
    type_col = col_map.get("type")
    if type_col is None:
        type_vals = np.full(n, "STAR", dtype=object)
    else:
        type_vals = catalog[type_col].astype(str).to_numpy(dtype=object)

    is_exp = np.array([_is_exp_type(v) for v in type_vals], dtype=bool)
    is_dev = np.array([_is_dev_type(v) for v in type_vals], dtype=bool)
    is_ser = np.array([_is_sersic_type(v) for v in type_vals], dtype=bool)
    source_type = np.full(n, SOURCE_STAR, dtype=np.int32)
    source_type[is_exp] = SOURCE_EXP
    source_type[is_dev] = SOURCE_DEV
    source_type[is_ser] = SOURCE_SERSIC

    # Sersic index seeds: catalog SERSIC_N column wins where finite; else
    # per-type defaults. EXP rows carry 1.0 but render from EXP_AMP/EXP_VAR.
    n_col = col_map.get("sersic_n")
    sersic_n = np.full(n, 1.0, dtype=float)
    sersic_n[is_dev] = DEFAULT_N_DEV
    sersic_n[is_ser] = DEFAULT_N_SERSIC
    if n_col is not None:
        n_raw = pd.to_numeric(catalog[n_col], errors="coerce").to_numpy(dtype=float)
        # Fixed-n source types IGNORE any input SERSIC_N: the index is
        # definitional (DEV -> n=4, EXP -> n=1), so letting a catalog value
        # override it would leak the seed into the rendered profile -- and
        # therefore into the fitted DEV/EXP FLUX (the amplitude is solved
        # against a profile whose shape is fixed by n). Only SERSIC rows,
        # whose index is a genuine free model parameter, take the catalog
        # seed here.
        use = np.isfinite(n_raw) & is_ser
        sersic_n[use] = n_raw[use]
        ignored = np.isfinite(n_raw) & (is_dev | is_exp)
        n_ignored = int(np.count_nonzero(ignored))
        if n_ignored:
            logger.warning(
                "%d DEV/EXP row(s) carry a SERSIC_N column value which is "
                "IGNORED (fixed-n source type: DEV keeps n=%.1f, EXP keeps "
                "n=1.0). The Sersic index is definitional for these types; "
                "only TYPE=SERSIC rows use the catalog SERSIC_N seed.",
                n_ignored, DEFAULT_N_DEV,
            )
    sersic_n = np.clip(sersic_n, SERSIC_N_MIN, SERSIC_N_MAX)

    re_col = col_map.get("re")
    if re_col is not None:
        re_raw = pd.to_numeric(catalog[re_col], errors="coerce").to_numpy(dtype=float)
        re = np.where(np.isfinite(re_raw), re_raw, float(re_fallback_pix))
    else:
        re = np.full(n, float(re_fallback_pix), dtype=float)
    re = np.clip(re, 0.3, 100.0)

    ell_col = col_map.get("ell")
    if ell_col is not None:
        ell = pd.to_numeric(catalog[ell_col], errors="coerce").to_numpy(dtype=float)
        ab = np.clip(1.0 - np.nan_to_num(ell, nan=0.0), 0.1, 1.0)
    else:
        ab = np.full(n, 0.8, dtype=float)

    theta_col = col_map.get("theta")
    if theta_col is not None:
        theta = pd.to_numeric(catalog[theta_col], errors="coerce").to_numpy(dtype=float)
        phi = _cpu_phi_from_theta(theta)
    else:
        phi = np.zeros(n, dtype=float)

    logre, ee1, ee2 = ellipse_soft_from_re_ab_phi_np(re, ab, phi)
    return source_type, logre, ee1, ee2, sersic_n.astype(np.float32)


def render_exp_profile_stamp_jax(H: int, W: int, logre, ee1, ee2):
    """Render a unit-flux intrinsic exponential profile stamp."""
    import jax.numpy as jnp

    yy = jnp.arange(H, dtype=jnp.float32) - jnp.float32(H // 2)
    xx = jnp.arange(W, dtype=jnp.float32) - jnp.float32(W // 2)
    y, x = jnp.meshgrid(yy, xx, indexing="ij")

    re = jnp.clip(jnp.exp(logre), 0.1, 1000.0)
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

    amps = jnp.asarray(EXP_AMP, dtype=jnp.float32)
    variances = jnp.asarray(EXP_VAR, dtype=jnp.float32)

    def one_component(amp, var):
        c00 = var * (a * a + b * b)
        c01 = var * (a * c + b * d)
        c11 = var * (c * c + d * d)
        det = jnp.maximum(c00 * c11 - c01 * c01, 1e-12)
        inv00 = c11 / det
        inv01 = -c01 / det
        inv11 = c00 / det
        r2 = inv00 * x * x + 2.0 * inv01 * x * y + inv11 * y * y
        norm = amp / (2.0 * jnp.pi * jnp.sqrt(det))
        return norm * jnp.exp(-0.5 * jnp.minimum(r2, 200.0))

    prof = jnp.sum(jax_vmap(one_component)(amps, variances), axis=0)
    total = jnp.sum(prof)
    return jnp.where(total > 0, prof / total, prof)


def jax_vmap(fn):
    import jax
    return jax.vmap(fn)


def render_source_stamp_at(
    psf_jax,
    x,
    y,
    grid_x0: int,
    grid_y0: int,
    source_type,
    logre,
    ee1,
    ee2,
    sersic_n=None,
    render_backend: str = "direct",
):
    """Render one source's unit-flux stamp at full-image position ``x,y``.

    ``render_backend`` is a trace-time constant:

    - ``"direct"`` (default): the historical path, byte-identical to the
      pre-extended-source code -- EXP via pixel-centre MoG + direct
      convolution + unit-BOX-sum renormalisation. STAR/EXP only; DEV/SERSIC
      sources must not reach this backend (enforced upstream).
    - ``"fft"``: extended sources via :mod:`sersic_mog` -- closed-form
      Gaussian-mixture Fourier transform x rfft2(PSF stamp), ANALYTIC total
      flux (no box renorm; the stamp integrates to 1 minus truncated wings).
      EXP renders from ``EXP_AMP/EXP_VAR``; DEV/SERSIC from the table at
      ``sersic_n``. STAR output is identical in both backends.
    """
    import jax.numpy as jnp
    from jax.scipy.signal import convolve

    raw_psf = evaluate_psf_at(psf_jax, x, y)
    H = int(psf_jax.H_psf)
    W = int(psf_jax.W_psf)

    if render_backend == "direct":
        gal = render_exp_profile_stamp_jax(H, W, logre, ee1, ee2)
        exp_stamp = convolve(gal, raw_psf, mode="same")
        exp_sum = jnp.sum(exp_stamp)
        exp_stamp = jnp.where(exp_sum > 0, exp_stamp / exp_sum, exp_stamp)
        raw = jnp.where(source_type == SOURCE_EXP, exp_stamp, raw_psf)
    elif render_backend == "fft":
        from . import sersic_mog as sg

        # ALL extended types render from the mixture table (EXP rows carry
        # sersic_n=1.0): total-flux definition = UNTRUNCATED Sersic for
        # n >= the table's splice (0.8), SDSS-truncated below (where the two
        # coincide to <0.5%). The legacy EXP_AMP/EXP_VAR constants are used
        # only by the "direct" backend.
        table = sg.get_table_jax()
        nn = jnp.float32(1.0) if sersic_n is None else sersic_n
        amps, vars_, core = sg.mog_params_at(table, nn)
        pad = sg.fft_pad_for(max(H, W))
        gal_stamp = sg.render_mog_stamp_fft(
            raw_psf, amps, vars_, core, logre, ee1, ee2, pad)
        raw = jnp.where(source_type == SOURCE_STAR, raw_psf, gal_stamp)
    else:  # pragma: no cover - config validation rejects other values
        raise ValueError(f"unknown render_backend: {render_backend!r}")

    x_grid = x - grid_x0
    y_grid = y - grid_y0
    ix_float = jnp.round(x_grid)
    iy_float = jnp.round(y_grid)
    dx = x_grid - ix_float
    dy = y_grid - iy_float
    return lanczos_shift_image(raw, dx, dy)


def render_star_stamps(psf_jax, pos_x, pos_y, grid_x0: int, grid_y0: int):
    """Unit-flux stamps for an ALL-STAR source set -- the exact STAR rows of
    :func:`render_source_stamps`, without the galaxy work.

    T6/G2 (2026-08-23, development ledger SS21 and notes): under the
    "direct" backend :func:`render_source_stamp_at` renders an EXP
    mixture-of-Gaussians profile and a ``H x W`` convolution for EVERY
    source and then discards both with ``jnp.where`` for STAR rows; under
    eager ``jax.vmap`` nothing can dead-code-eliminate that, so a star-only
    grid paid it twice per band (blank-response + error model) and compiled
    ~49 of its ~113 per-grid eager programs for it. This is the same exact
    star-only path the island scene-subtract shipped as T3 v2: the
    primitives below are the ones the STAR rows already execute, in the
    same order, on the same batch (``jnp.where`` selects ``raw_psf``
    bit-for-bit), including the GRID-LOCAL rounding of
    :func:`render_source_stamp_at` -- so the stamps are bitwise identical
    to the full chain's STAR rows (verified CPU 0-ulp incl. non-zero grid
    origins; identical executables). Callers must dispatch on the host-known
    source types (every source STAR); mixed grids keep the full chain.
    """
    import jax
    import jax.numpy as jnp

    def _one(x, y):
        raw_psf = evaluate_psf_at(psf_jax, x, y)
        x_grid = x - grid_x0
        y_grid = y - grid_y0
        ix_float = jnp.round(x_grid)
        iy_float = jnp.round(y_grid)
        dx = x_grid - ix_float
        dy = y_grid - iy_float
        return lanczos_shift_image(raw_psf, dx, dy)

    return jax.vmap(_one)(pos_x, pos_y)


def render_source_stamps(
    psf_jax,
    pos_x,
    pos_y,
    grid_x0: int,
    grid_y0: int,
    source_type,
    logre,
    ee1,
    ee2,
    sersic_n=None,
    render_backend: str = "direct",
):
    """Render unit-flux stamps for mixed star/galaxy sources."""
    import jax
    import jax.numpy as jnp
    if sersic_n is None:
        sersic_n = jnp.ones_like(jnp.asarray(logre))
    return jax.vmap(
        lambda x, y, t, lr, e1, e2, sn: render_source_stamp_at(
            psf_jax, x, y, grid_x0, grid_y0, t, lr, e1, e2,
            sersic_n=sn, render_backend=render_backend,
        )
    )(pos_x, pos_y, source_type, logre, ee1, ee2, sersic_n)


__all__ = [
    "SOURCE_STAR",
    "SOURCE_EXP",
    "SOURCE_DEV",
    "SOURCE_SERSIC",
    "EXP_AMP",
    "EXP_VAR",
    "is_supported_source_type",
    "catalog_source_params",
    "ellipse_soft_from_re_ab_phi_np",
    "ellipse_soft_to_re_ab_phi_np",
    "render_exp_profile_stamp_jax",
    "render_source_stamp_at",
    "render_source_stamps",
    "render_star_stamps",
]
