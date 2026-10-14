"""Sparse-stamp linear forced-photometry for grid-scale fits.

Replaces :mod:`.forced_phot`'s dense ``(n_sources, n_grid_pixels)``
projection (which is fine for synthetic tests but blows memory at
T23405 grid scale = 750 sources x 13 M pixels x 4 B = 39 GB per band)
with a SPARSE-STAMP accumulation:

- Each source contributes ONLY to its ~55x55 stamp footprint.
- Source-source coupling ``T_s^T W T_t`` is exactly 0 unless the two
  stamp bboxes overlap; we precompute a per-source neighbor list
  (padded to a fixed ``k_max`` size for vmap) and only evaluate the
  inner product for actual neighbors.
- Source-background coupling ``T_s^T W B_m`` is per-source local
  (depends only on pixels under the source's stamp).
- Background-background block ``B_m^T W B_n`` is a full-grid
  weighted Gram matrix; it's tiny per-band (M x M = 6x6 for the
  default quadratic basis) and computed via streaming chunked sums
  to bound peak memory.

Memory profile (per band, T23405-class grid):

- Source stamp tensor: ``n_sources x H_psf x W_psf x 4 B`` ~ 9 MB
- Per-source bg basis under stamp: ``n_sources x H_psf x W_psf x M
  x 4 B`` ~ 54 MB
- Source-source pairs: ``n_sources x k_max`` neighbor indices +
  scalar inner products ~ 1.5 MB
- Bg full-grid streaming Gram: ``M x M`` ~ negligible
- Final AtWA: ``(n_sources + M) x (n_sources + M)`` x 4 B ~ 2.2 MB

Total: ~70 MB peak per band. 20 bands processed sequentially fits
comfortably in 24 GB of GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree

from ..optimize.gpu.kernels.lanczos import lanczos_shift_image
from .bg_polynomial import (
    N_TERMS_CONST, N_TERMS_LINEAR, N_TERMS_QUADRATIC,
    make_bg_basis_jax, n_terms_for_order,
)
from .psf_polynomial import (
    PsfPolynomial, PsfPolynomialJax,
    evaluate_psf_at, evaluate_psf_at_batch,
)
from .source_model import SOURCE_EXP, SOURCE_STAR, render_source_stamps


# Default neighbor cap. Padded to this size for JIT-friendly vmap.
# For T23405-class crowded fields (~3 sources per arcmin^2 with
# ~55-px PSF), 30 neighbors per source is generous.
DEFAULT_K_MAX = 32


@dataclass
class GridBandInputs:
    """Per-grid per-band data for the sparse forced-phot solve.

    Shapes:

    - ``data``, ``invvar``, ``pix_good``: (band_h, band_w)
    - ``pos_x``, ``pos_y``: (n_sources,) FULL-image pixel coords
      (the polynomial PSF expects FULL-image coords; the stamp
      placement converts to grid-local via ``grid_x0/grid_y0``).
    - ``psf_jax``: per-band polynomial PSF model.
    - ``grid_x0``, ``grid_y0``: top-left of THIS grid in full image.
    - ``bg_x_origin/scale, bg_y_origin/scale``: bg polynomial
      normalization (reuse PSF normalization for consistency).
    """
    data: jnp.ndarray
    invvar: jnp.ndarray
    pix_good: jnp.ndarray
    pos_x: jnp.ndarray
    pos_y: jnp.ndarray
    psf_jax: PsfPolynomialJax
    band_h: int
    band_w: int
    grid_x0: int
    grid_y0: int
    bg_x_origin: float
    bg_x_scale: float
    bg_y_origin: float
    bg_y_scale: float
    # Shared per-grid basis. It depends only on grid geometry/background order,
    # so all bands in a grid can reuse the same device array.
    bg_basis_flat: jnp.ndarray | None = None
    # Source model arrays. 0=star, 1=ExpGalaxy, 2=DevGalaxy, 3=SersicGalaxy.
    source_type: jnp.ndarray | None = None
    logre: jnp.ndarray | None = None
    ee1: jnp.ndarray | None = None
    ee2: jnp.ndarray | None = None
    sersic_n: jnp.ndarray | None = None
    # Trace-time renderer choice ("direct" = historical path, "fft" =
    # analytic-flux sersic_mog path). Part of the traced program, so grids
    # differing here compile separately.
    render_backend: str = "direct"


@dataclass
class SparseForcedPhotResult:
    """Per-band fit result from :func:`solve_band_sparse`."""
    fluxes: np.ndarray         # (n_sources,)
    bg_coefs: np.ndarray       # (N_TERMS_QUADRATIC,)
    flux_var: np.ndarray       # (n_sources,) diag of inv(AtWA)
    bg_var: np.ndarray         # (N_TERMS_QUADRATIC,)
    converged: bool


# ---------------------------------------------------------------------------
# Stamp rendering (Lanczos-shifted, for differentiable position fits)
# ---------------------------------------------------------------------------


_RENDER_STAMPS_JIT_CACHE: dict[tuple, "jax.stages.Wrapped"] = {}


def _make_render_stamps_jit(n_terms: int, renorm: bool):
    """Build ONE jitted render program for a (n_terms, renorm) family.

    The eager path re-traced two fresh-lambda vmaps on EVERY call
    (~50-100 ms of Python dispatch per band per linear solve -- measured
    as the dominant host cost of the per-grid fit). Tracing once per
    shape set and persisting the compiled program (the JIT cache
    threshold is 0 in workers) removes that dispatch entirely. Same
    operations in the same order as the eager body; XLA fusion may
    differ at the rounding level (trajectory-class change, full-gate
    validated)."""
    from .psf_polynomial import _spatial_basis_jax

    def _body(coefs, x_origin, x_scale, y_origin, y_scale,
              pos_x, pos_y, grid_x0, grid_y0):
        def _one(x, y):
            X = (x - x_origin) / x_scale
            Y = (y - y_origin) / y_scale
            basis = _spatial_basis_jax(X, Y, n_terms=n_terms)
            # PRECISION pin (development ledger SS23-32 + SS24): this
            # contraction BUILDS the spatially varying PSF stamp, upstream of
            # every render and every flux, and forced photometry never
            # renormalises the stamp afterwards -- so a truncated mantissa
            # here is a multiplicative error on every measured flux. XLA's
            # default demotes it to TF32 (10-bit mantissa). n_terms is 15, so
            # the contraction is trivial and HIGHEST is free.
            return jnp.tensordot(coefs, basis, axes=([-1], [-1]),
                                 precision=jax.lax.Precision.HIGHEST)

        raw = jax.vmap(_one)(pos_x, pos_y)
        if renorm:
            sums = jnp.sum(raw, axis=(1, 2), keepdims=True)
            raw = raw / jnp.maximum(sums, jnp.float32(1e-12))
        x_grid = pos_x - grid_x0
        y_grid = pos_y - grid_y0
        ix_float = jnp.round(x_grid)
        iy_float = jnp.round(y_grid)
        dx = x_grid - ix_float
        dy = y_grid - iy_float
        return jax.vmap(lambda s, dxi, dyi: lanczos_shift_image(s, dxi, dyi))(
            raw, dx, dy,
        )

    return jax.jit(_body)


def render_stamps(
    psf_jax: PsfPolynomialJax,
    pos_x: jnp.ndarray, pos_y: jnp.ndarray,
    grid_x0: int, grid_y0: int,
) -> jnp.ndarray:
    """Render ``(n_sources, H_psf, W_psf)`` Lanczos-sub-pixel-shifted
    PSF stamps for all sources. Smooth in (x, y) so position fits
    have non-zero gradient even from the placement.

    Unit-sum renormalization (env TRACT7DT_RENDER_STAMP_RENORM=0 to
    disable): every TRAINING stamp sums to exactly 1, but the per-pixel
    polynomial EXTRAPOLATES between/beyond training stars, letting the
    rendered box sum drift across the field (measured on T08285:
    0.978-1.040). That drift multiplies straight into every fitted flux
    as a spurious spatial ZP term; renormalizing restores the training
    convention exactly.
    """
    renorm = os.environ.get("TRACT7DT_RENDER_STAMP_RENORM", "1") != "0"
    eager = os.environ.get("TRACT7DT_RENDER_STAMPS_JIT", "1") == "0"
    if (not eager) and type(psf_jax).__name__ == "PsfPolynomialJax":
        n_terms = int(psf_jax.coefs.shape[-1])
        key = (n_terms, bool(renorm))
        fn = _RENDER_STAMPS_JIT_CACHE.get(key)
        if fn is None:
            fn = _make_render_stamps_jit(n_terms, bool(renorm))
            _RENDER_STAMPS_JIT_CACHE[key] = fn
        return fn(
            psf_jax.coefs,
            jnp.float32(psf_jax.x_origin), jnp.float32(psf_jax.x_scale),
            jnp.float32(psf_jax.y_origin), jnp.float32(psf_jax.y_scale),
            pos_x, pos_y,
            jnp.float32(grid_x0), jnp.float32(grid_y0),
        )

    # Eager fallback: the
    # TRACT7DT_RENDER_STAMPS_JIT=0 escape hatch.
    raw_stamps = evaluate_psf_at_batch(psf_jax, pos_x, pos_y)
    if renorm:
        sums = jnp.sum(raw_stamps, axis=(1, 2), keepdims=True)
        raw_stamps = raw_stamps / jnp.maximum(sums, jnp.float32(1e-12))
    x_grid = pos_x - grid_x0
    y_grid = pos_y - grid_y0
    ix_float = jnp.round(x_grid)
    iy_float = jnp.round(y_grid)
    dx = x_grid - ix_float
    dy = y_grid - iy_float
    return jax.vmap(lambda s, dxi, dyi: lanczos_shift_image(s, dxi, dyi))(
        raw_stamps, dx, dy,
    )


def _source_arrays_for_inputs(inputs: GridBandInputs):
    n_sources = int(inputs.pos_x.shape[0])
    source_type = inputs.source_type
    if source_type is None:
        source_type = jnp.full((n_sources,), SOURCE_STAR, dtype=jnp.int32)
    logre = inputs.logre
    if logre is None:
        logre = jnp.zeros((n_sources,), dtype=jnp.float32)
    ee1 = inputs.ee1
    if ee1 is None:
        ee1 = jnp.zeros((n_sources,), dtype=jnp.float32)
    ee2 = inputs.ee2
    if ee2 is None:
        ee2 = jnp.zeros((n_sources,), dtype=jnp.float32)
    sersic_n = inputs.sersic_n
    if sersic_n is None:
        sersic_n = jnp.ones((n_sources,), dtype=jnp.float32)
    return source_type, logre, ee1, ee2, sersic_n


def _has_exp_sources(inputs: GridBandInputs) -> bool:
    """True when the grid holds ANY extended source (EXP/DEV/SERSIC) --
    i.e. the mixed renderer must be used instead of the star-only fast path.
    (Name kept for call-site stability; semantics widened 2026-07.)"""
    if inputs.source_type is None:
        return False
    try:
        return bool(np.any(np.asarray(inputs.source_type) != SOURCE_STAR))
    except Exception:
        return True


def stamp_top_left(
    pos_x: jnp.ndarray, pos_y: jnp.ndarray,
    grid_x0: int, grid_y0: int,
    H_psf: int, W_psf: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integer top-left placement of each source's stamp in grid-local
    coords. Returns ``(ix_top, iy_top)`` arrays of shape ``(n_sources,)``."""
    ix = jnp.round(pos_x - grid_x0).astype(jnp.int32) - W_psf // 2
    iy = jnp.round(pos_y - grid_y0).astype(jnp.int32) - H_psf // 2
    return ix, iy


# ---------------------------------------------------------------------------
# Neighbor list (host-side, computed once per stage)
# ---------------------------------------------------------------------------


def build_neighbor_list(
    pos_x: np.ndarray, pos_y: np.ndarray,
    H_psf: int, W_psf: int,
    k_max: int = DEFAULT_K_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed-size neighbor list per source.

    Two sources's stamps overlap iff their stamp bboxes overlap. For
    same-size stamps (H_psf x W_psf), the bboxes overlap iff:

        |dx| < W_psf  AND  |dy| < H_psf

    (centers within (W_psf, H_psf) in each axis). We use this as the
    neighbor predicate.

    Returns
    -------
    neighbor_idx : np.ndarray of int32, shape (n_sources, k_max)
        ``neighbor_idx[s, k]`` is the index of the k-th neighbor of
        source s. Padded to ``k_max`` with sentinel ``-1`` for slots
        without an actual neighbor.
    neighbor_mask : np.ndarray of bool, shape (n_sources, k_max)
        True where neighbor_idx is a real neighbor.
    """
    n = pos_x.shape[0]
    neighbor_idx = np.full((n, k_max), -1, dtype=np.int32)
    neighbor_mask = np.zeros((n, k_max), dtype=bool)

    pos_x = np.asarray(pos_x, dtype=np.float64)
    pos_y = np.asarray(pos_y, dtype=np.float64)
    overlap_dx = float(W_psf)
    overlap_dy = float(H_psf)

    finite = np.isfinite(pos_x) & np.isfinite(pos_y)
    if np.any(finite):
        xy = np.column_stack([pos_x[finite], pos_y[finite]])
        finite_idx = np.where(finite)[0]
        tree = cKDTree(xy)
        # Query a circumscribed radius, then keep the exact rectangular
        # stamp-overlap predicate below.
        query_radius = float(np.hypot(overlap_dx, overlap_dy))
        candidate_lists = tree.query_ball_point(xy, r=query_radius)
    else:
        finite_idx = np.zeros(0, dtype=np.int64)
        candidate_lists = []

    for local_s, s in enumerate(finite_idx):
        cand = finite_idx[np.asarray(candidate_lists[local_s], dtype=np.int64)]
        cand = cand[cand != s]
        if cand.size == 0:
            continue
        dx = np.abs(pos_x[cand] - pos_x[s])
        dy = np.abs(pos_y[cand] - pos_y[s])
        nbr_indices = cand[(dx < overlap_dx) & (dy < overlap_dy)]
        if nbr_indices.size > k_max:
            # Keep the k_max closest.
            d2 = (pos_x[nbr_indices] - pos_x[s]) ** 2 + (pos_y[nbr_indices] - pos_y[s]) ** 2
            keep = np.argsort(d2)[:k_max]
            nbr_indices = nbr_indices[keep]
        nk = nbr_indices.size
        neighbor_idx[s, :nk] = nbr_indices.astype(np.int32)
        neighbor_mask[s, :nk] = True
    return neighbor_idx, neighbor_mask


# ---------------------------------------------------------------------------
# JIT-compiled sparse AtWA / AtWb accumulator (per band)
# ---------------------------------------------------------------------------


def _make_solve_band_jit(
    H_psf: int, W_psf: int,
    band_h: int, band_w: int,
    n_sources: int, k_max: int,
    M: int,  # bg basis terms
):
    """Build + jit the per-band sparse forced-phot solver.

    Closure captures all the static shapes; the runtime inputs are
    only the actual data arrays (data, invvar, stamps, etc.).
    """

    def per_source_self_blocks(stamps, w_under, d_under, bg_under):
        """Per-source diagonal entries of ``T^T W T``, ``T^T W d``,
        and the source-bg coupling row ``T_s^T W B``.

        Vectorized over sources via vmap.
        """
        # stamps: (n, H_psf, W_psf)
        # w_under: (n, H_psf, W_psf)
        # d_under: (n, H_psf, W_psf)
        # bg_under: (n, H_psf, W_psf, M)
        # Returns:
        #   TtWT_diag: (n,)
        #   TtWd: (n,)
        #   TtWB: (n, M)
        wstamps = w_under * stamps
        TtWT_diag = jnp.sum(wstamps * stamps, axis=(1, 2))
        TtWd = jnp.sum(wstamps * d_under, axis=(1, 2))
        TtWB = jnp.sum(wstamps[..., None] * bg_under, axis=(1, 2))
        return TtWT_diag, TtWd, TtWB

    def per_pair_overlap(s_idx, t_idx, mask, stamps, w_under,
                         ix_top, iy_top):
        """Compute T_s^T W T_t for one (s, t) pair where t is a
        neighbor of s. Returns scalar (= contribution to AtWA[s, t]).

        Implementation: gather the t-stamp values at s-stamp's pixel
        locations (via index arithmetic), then weighted dot product
        with s-stamp's values.
        """
        # Stamp s pixel (ky, kx) is at grid coord (iy_top[s]+ky,
        # ix_top[s]+kx). To find the corresponding pixel in stamp t,
        # we map that to t's local coord:
        #   t_ky = (grid_y - iy_top[t]) = ky + (iy_top[s] - iy_top[t])
        #   t_kx = (grid_x - ix_top[t]) = kx + (ix_top[s] - ix_top[t])
        # If (t_ky, t_kx) is in [0, H_psf) x [0, W_psf), they overlap.
        di = iy_top[s_idx] - iy_top[t_idx]  # scalar
        dj = ix_top[s_idx] - ix_top[t_idx]  # scalar

        ky = jnp.arange(H_psf)
        kx = jnp.arange(W_psf)
        t_ky = ky[:, None] + di
        t_kx = kx[None, :] + dj
        in_t = (t_ky >= 0) & (t_ky < H_psf) & (t_kx >= 0) & (t_kx < W_psf)
        safe_t_ky = jnp.clip(t_ky, 0, H_psf - 1)
        safe_t_kx = jnp.clip(t_kx, 0, W_psf - 1)
        t_stamp_at_s_pixels = jnp.where(in_t, stamps[t_idx, safe_t_ky, safe_t_kx], 0.0)
        # Weighted inner product over s's pixels.
        contrib = jnp.sum(w_under[s_idx] * stamps[s_idx] * t_stamp_at_s_pixels)
        return contrib * mask.astype(jnp.float32)  # zero if not a real neighbor

    def per_source_neighbors(s_idx, neighbor_idx_row, neighbor_mask_row,
                             stamps, w_under, ix_top, iy_top):
        """For source s, compute its ROW of TtWT off-diagonals (k_max
        scalars) - one per slot in the neighbor list.

        Returns a (k_max,) array of inner products.
        """
        return jax.vmap(
            lambda t, m: per_pair_overlap(
                s_idx, t, m, stamps, w_under, ix_top, iy_top,
            )
        )(neighbor_idx_row, neighbor_mask_row)

    def bg_full_grid_gram_and_target(bg_basis_flat, w_flat, d_flat):
        """Bg-bg Gram block (M x M) + Bg-data vector (M,) over the full
        grid. Done via direct matmul; size dominated by ``n_pix * M``
        which is bounded for our grids."""
        # bg_basis_flat: (n_pix, M)
        # w_flat: (n_pix,)
        # d_flat: (n_pix,)
        bw = bg_basis_flat * w_flat[:, None]  # (n_pix, M)
        # PRECISION, pinned rather than inherited (development ledger SS23-31
        # + SS24). These two are the only reductions in the engine that run
        # over a whole grid, and XLA's default lowers an f32 dot to TF32 --
        # 10 mantissa bits instead of 23. It reaches this site through cuBLAS,
        # not inline PTX: the shipped sparse cache carries 41 `__cublas$gemm`
        # calls at `"operand_precision":["DEFAULT","DEFAULT"]` and **37 of them
        # are in `jit_solver_inner`**, i.e. here. Measured on the real sparse
        # anchor, turning the demotion off moved this background plane by
        # 5.3e-4 ADU (median) / 2.1e-2 (max) against a typical coefficient of
        # 0.12 ADU, and the plane then reproduced BIT-IDENTICALLY across cold
        # runs (176/176 coefficients, vs 64.6% before).
        # M is 3 at the shipped bg_poly order 1, so this is 9*n_pix
        # multiply-adds -- bandwidth-bound, and HIGHEST costs no extra traffic.
        # The global default set at every x64 site already implies this; the
        # explicit pin keeps the site correct if an operator overrides that.
        _P = jax.lax.Precision.HIGHEST
        BtWB = jnp.matmul(bw.T, bg_basis_flat, precision=_P)   # (M, M)
        BtWd = jnp.matmul(bw.T, d_flat, precision=_P)          # (M,)
        return BtWB, BtWd

    def assemble_and_solve(
        TtWT_diag, TtWT_off, neighbor_idx, neighbor_mask,
        TtWd, TtWB, BtWB, BtWd, reg,
    ):
        """Assemble the full (n_sources + M) x (n_sources + M) AtWA,
        regularize, solve in FP64, return (fluxes, bg_coefs, flux_var,
        bg_var)."""
        N = n_sources
        Ntot = N + M
        AtWA = jnp.zeros((Ntot, Ntot), dtype=jnp.float32)
        # Diagonal source block.
        AtWA = AtWA.at[jnp.arange(N), jnp.arange(N)].set(TtWT_diag)
        # Off-diagonal source-source block via scatter. Each pair
        # (s, t) appears TWICE in the neighbor list (once as (s, t)
        # and once as (t, s)), so the (s, t) iteration fills
        # AtWA[s, t] and the (t, s) iteration fills AtWA[t, s].
        # We do NOT also scatter the transpose, and we do NOT
        # symmetrize afterward - that would double-count.
        flat_s = jnp.repeat(jnp.arange(N), k_max)
        flat_t = neighbor_idx.reshape(-1)
        flat_m = neighbor_mask.reshape(-1)
        flat_v = TtWT_off.reshape(-1)
        safe_t = jnp.where(flat_m, flat_t, 0).astype(jnp.int32)
        safe_v = jnp.where(flat_m, flat_v, 0.0)
        AtWA = AtWA.at[flat_s, safe_t].add(safe_v)
        # Source-bg block (N x M).
        AtWA = AtWA.at[:N, N:].set(TtWB)
        AtWA = AtWA.at[N:, :N].set(TtWB.T)
        # Bg-bg block (M x M).
        AtWA = AtWA.at[N:, N:].set(BtWB)

        AtWb = jnp.concatenate([TtWd, BtWd])

        # ELEMENT-WISE ridge: reg * diag(AtWA) added to each diagonal.
        # The source-source (TtWT) and background-background (BtWB)
        # blocks live on VERY different magnitude scales (source diag
        # ~ stamp_pix * sigma^-2 ~ 1e-7; bg diag ~ n_full_pix * sigma^-2
        # ~ 1e3, so ~10 orders of magnitude apart). A scalar ridge =
        # reg * diag_avg would be dominated by the bg diagonal and
        # inflate the source diagonal by orders of magnitude, shrinking
        # all source fluxes catastrophically. Element-wise ridge keeps
        # each block's conditioning at the same relative level.
        diag = jnp.diag(AtWA)
        ridge_diag = reg * jnp.maximum(jnp.abs(diag), 1e-30)
        AtWA_reg = AtWA + jnp.diag(ridge_diag)

        AtWA64 = AtWA_reg.astype(jnp.float64)
        AtWb64 = AtWb.astype(jnp.float64)
        x = jnp.linalg.solve(AtWA64, AtWb64).astype(jnp.float32)
        # Variance = diag of inverse.
        inv = jnp.linalg.inv(AtWA64).astype(jnp.float32)
        var_diag = jnp.diag(inv)
        return x[:N], x[N:], var_diag[:N], var_diag[N:]

    def solver_inner(
        data, invvar, pix_good,
        stamps, ix_top, iy_top,
        neighbor_idx, neighbor_mask,
        bg_basis_flat,
        reg,
    ):
        # Per-source pixel coordinates in grid frame.
        ky = jnp.arange(H_psf)
        kx = jnp.arange(W_psf)
        ys = iy_top[:, None, None] + ky[None, :, None]
        xs = ix_top[:, None, None] + kx[None, None, :]
        in_bounds = (
            (ys >= 0) & (ys < band_h) & (xs >= 0) & (xs < band_w)
        )
        safe_ys = jnp.clip(ys, 0, band_h - 1)
        safe_xs = jnp.clip(xs, 0, band_w - 1)
        w_grid = jnp.where(pix_good, invvar, 0.0)
        # Sample under each source's stamp.
        w_under = jnp.where(in_bounds, w_grid[safe_ys, safe_xs], 0.0)
        d_under = data[safe_ys, safe_xs]
        # Bg basis under each stamp.
        # bg_basis_flat is (n_pix, M); reshape to (band_h, band_w, M)
        # then gather.
        bg_basis_grid = bg_basis_flat.reshape(band_h, band_w, M)
        bg_under = bg_basis_grid[safe_ys, safe_xs, :]

        # Mask source stamps to in-bounds (out-of-bounds pixels
        # contribute 0 from a clipped value, but we want exact zero).
        stamps_eff = jnp.where(in_bounds, stamps, 0.0)

        # Per-source self / source-bg / source-data contributions.
        TtWT_diag, TtWd, TtWB = per_source_self_blocks(
            stamps_eff, w_under, d_under, bg_under,
        )

        # Per-source off-diagonal source-source.
        TtWT_off = jax.vmap(
            lambda s, ni, nm: per_source_neighbors(
                s, ni, nm, stamps_eff, w_under, ix_top, iy_top,
            )
        )(jnp.arange(n_sources), neighbor_idx, neighbor_mask)

        # Bg-bg full-grid block.
        w_flat = w_grid.reshape(-1)
        d_flat = data.reshape(-1).astype(jnp.float32)
        BtWB, BtWd = bg_full_grid_gram_and_target(bg_basis_flat, w_flat, d_flat)

        return assemble_and_solve(
            TtWT_diag, TtWT_off, neighbor_idx, neighbor_mask,
            TtWd, TtWB, BtWB, BtWd, reg,
        )

    # Return BOTH the un-jit'd solver_inner (so we can vmap it cleanly)
    # AND the jit'd version (the per-band path).
    return solver_inner, jax.jit(solver_inner)


# ---------------------------------------------------------------------------
# Top-level entry: per-band sparse forced-phot solve.
# ---------------------------------------------------------------------------


_SOLVER_CACHE: dict[tuple, "jax.stages.Wrapped"] = {}
_BATCHED_SOLVER_CACHE: dict[tuple, "jax.stages.Wrapped"] = {}


class _PendingBandSolve:
    """In-flight per-band solve: device handles, not yet synced to host.

    Produced by :func:`launch_band_sparse`, finalized by
    :func:`collect_band_sparse`. ``stamps`` keeps the rendered per-source
    stamp DeviceArray alive so callers may reuse it (the position step
    runs at the SAME positions as its iteration's linear solve)."""
    __slots__ = ("fluxes", "bg_coefs", "flux_var", "bg_var", "stamps")

    def __init__(self, fluxes, bg_coefs, flux_var, bg_var, stamps):
        self.fluxes = fluxes
        self.bg_coefs = bg_coefs
        self.flux_var = flux_var
        self.bg_var = bg_var
        self.stamps = stamps


def launch_band_sparse(
    inputs: GridBandInputs,
    neighbor_idx: np.ndarray,
    neighbor_mask: np.ndarray,
    reg: float = 1e-9,
    bg_poly_order: int = 2,
    nbr_idx_jax=None,
    nbr_mask_jax=None,
    reg_jax=None,
) -> _PendingBandSolve:
    """Render + dispatch one band's sparse solve WITHOUT a host sync.

    Same XLA programs, same inputs, same single-stream submission order as
    the synchronous :func:`solve_band_sparse` -- only the host sync point
    moves to :func:`collect_band_sparse`. This lets the per-band loop
    overlap band b's host-side render (~100 ms of eager-vmap dispatch, the
    dominant per-band cost) with band b-1's GPU solve instead of draining
    the pipeline with a per-band ``np.asarray`` (measured ~0.9 s per
    20-band stage1 call; bit-identical by digest at production shapes).

    ``nbr_idx_jax``/``nbr_mask_jax``/``reg_jax`` accept pre-uploaded device
    constants so a 20-band loop uploads them once instead of per band.
    """
    M = n_terms_for_order(bg_poly_order)
    H_psf = inputs.psf_jax.H_psf
    W_psf = inputs.psf_jax.W_psf
    band_h = inputs.band_h
    band_w = inputs.band_w
    n_sources = int(inputs.pos_x.shape[0])
    k_max = int(neighbor_idx.shape[1])

    cache_key = (band_h, band_w, n_sources, H_psf, W_psf, k_max, M)
    fn = _SOLVER_CACHE.get(cache_key)
    if fn is None:
        _, fn = _make_solve_band_jit(
            H_psf=H_psf, W_psf=W_psf,
            band_h=band_h, band_w=band_w,
            n_sources=n_sources, k_max=k_max, M=M,
        )
        _SOLVER_CACHE[cache_key] = fn

    if _has_exp_sources(inputs):
        # Render mixed source stamps with Lanczos sub-pixel shift.
        stamps = render_source_stamps(
            inputs.psf_jax, inputs.pos_x, inputs.pos_y,
            inputs.grid_x0, inputs.grid_y0,
            *_source_arrays_for_inputs(inputs),
            render_backend=inputs.render_backend,
        )
    else:
        # Preserve the original point-source path exactly for all-star inputs.
        stamps = render_stamps(
            inputs.psf_jax, inputs.pos_x, inputs.pos_y,
            inputs.grid_x0, inputs.grid_y0,
        )
    ix_top, iy_top = stamp_top_left(
        inputs.pos_x, inputs.pos_y,
        inputs.grid_x0, inputs.grid_y0, H_psf, W_psf,
    )
    bg_basis_flat = inputs.bg_basis_flat
    if bg_basis_flat is None:
        bg_basis_flat = make_bg_basis_jax(
            band_h, band_w, inputs.grid_x0, inputs.grid_y0,
            inputs.bg_x_origin, inputs.bg_x_scale,
            inputs.bg_y_origin, inputs.bg_y_scale,
            order=bg_poly_order,
        )

    if nbr_idx_jax is None:
        nbr_idx_jax = jnp.asarray(neighbor_idx)
    if nbr_mask_jax is None:
        nbr_mask_jax = jnp.asarray(neighbor_mask)
    if reg_jax is None:
        reg_jax = jnp.asarray(reg, dtype=jnp.float32)

    fluxes, bg_coefs, flux_var, bg_var = fn(
        inputs.data, inputs.invvar, inputs.pix_good,
        stamps, ix_top, iy_top,
        nbr_idx_jax, nbr_mask_jax,
        bg_basis_flat,
        reg_jax,
    )
    return _PendingBandSolve(fluxes, bg_coefs, flux_var, bg_var, stamps)


def collect_band_sparse(pending: _PendingBandSolve) -> SparseForcedPhotResult:
    """Sync one in-flight band solve to host (single batched fetch)."""
    fluxes, bg_coefs, flux_var, bg_var = jax.device_get(
        (pending.fluxes, pending.bg_coefs, pending.flux_var, pending.bg_var))
    finite = bool(np.all(np.isfinite(fluxes)) and
                  np.all(np.isfinite(bg_coefs)))
    return SparseForcedPhotResult(
        fluxes=fluxes,
        bg_coefs=bg_coefs,
        flux_var=flux_var,
        bg_var=bg_var,
        converged=finite,
    )


def solve_band_sparse(
    inputs: GridBandInputs,
    neighbor_idx: np.ndarray,
    neighbor_mask: np.ndarray,
    reg: float = 1e-9,
    bg_poly_order: int = 2,
) -> SparseForcedPhotResult:
    """Solve fluxes + bg coefs for one band of one grid via sparse-stamp
    accumulation. Synchronous wrapper over launch/collect (see
    :func:`launch_band_sparse` for the pipelined multi-band form).

    Caller is responsible for providing the neighbor list (typically
    built once per stage via :func:`build_neighbor_list`).

    ``bg_poly_order`` selects the polynomial spatial order for the
    background:

    - 0: constant (1 coef), like per-patch CPU sky scalar
    - 1: linear plane (3 coefs: 1, x, y)
    - 2: quadratic (6 coefs: 1, x, y, x^2, xy, y^2)

    JIT-cached on (band_h, band_w, n_sources, H_psf, W_psf, k_max, M).
    """
    return collect_band_sparse(launch_band_sparse(
        inputs, neighbor_idx, neighbor_mask,
        reg=reg, bg_poly_order=bg_poly_order,
    ))


__all__ = [
    "DEFAULT_K_MAX",
    "GridBandInputs",
    "SparseForcedPhotResult",
    "build_neighbor_list",
    "render_stamps",
    "render_source_stamps",
    "stamp_top_left",
    "solve_band_sparse",
    "launch_band_sparse",
    "collect_band_sparse",
]
