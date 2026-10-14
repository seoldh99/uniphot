"""Joint multi-source LSQR / CG-on-normal-equations position step.

This is the JOINT alternative to ``stage2_source_newton_step``'s
per-source Gauss-Seidel Newton.  It assembles a single residual function
for ALL sources in the grid and lets a Krylov solver (currently
``jax.scipy.sparse.linalg.cg`` on the normal equations) take a step
that accounts for the full off-diagonal Hessian coupling between
overlapping sources.

The flux + bg subsystem is still solved via the existing closed-form
Cholesky in ``stage1_solve``; only the nonlinear position + shape
parameters are handled here.

Public entry point: :func:`joint_position_step`.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .source_model import (
    SOURCE_EXP, SOURCE_STAR, render_source_stamps, sersic_u_to_n,
)
from .forced_phot_sparse import _make_solve_band_jit, stamp_top_left
from .bg_polynomial import n_terms_for_order


# ---------------------------------------------------------------------------
# Per-grid data pytrees.
#
# jax.jit bakes CLOSED-OVER arrays into the compiled program as
# compile-time CONSTANTS. The joint/varpro step functions are cached in
# solver_grid keyed only on SHAPES, so a residual closure that captured
# the per-grid data stacks made a second same-shape grid silently reuse
# the first grid's compiled program -- i.e. fit against grid 1's data.
# All per-grid arrays therefore flow through the jitted step as TRACED
# arguments via the NamedTuples below (NamedTuples are JAX pytrees);
# only true statics (shapes, the render_backend string, and the
# per-band PSF models, which are identical across grids in a run and
# whose H/W is part of the cache key) stay in the closure.
# ---------------------------------------------------------------------------


class JointArrays(NamedTuple):
    """Per-grid data for the frozen-flux joint residual
    (:func:`make_residual_fn`). Traced pytree argument of the step."""
    data_stack: jnp.ndarray       # (n_bands, H, W)
    invvar_stack: jnp.ndarray     # (n_bands, H, W)
    pix_good_stack: jnp.ndarray   # (n_bands, H, W) bool
    bg_basis_grid: jnp.ndarray    # (H, W, M)
    bg_coefs: jnp.ndarray         # (n_bands, M)  -- frozen during pos step
    fluxes: jnp.ndarray           # (n_sources, n_bands)  -- frozen
    source_type: jnp.ndarray      # (n_sources,) int32 (SOURCE_* codes)
    sersic_n: jnp.ndarray         # (n_sources,) float32
    grid_x0: jnp.ndarray          # () float32 -- per-grid image origin
    grid_y0: jnp.ndarray          # () float32


class VarproArrays(NamedTuple):
    """Per-grid data for the variable-projection residual
    (:func:`make_varpro_residual_fn`). Traced pytree argument of the step."""
    data_stack: jnp.ndarray       # (n_bands, H, W)
    invvar_stack: jnp.ndarray     # (n_bands, H, W)
    pix_good_stack: jnp.ndarray   # (n_bands, H, W) bool
    bg_basis_grid: jnp.ndarray    # (H, W, M)
    source_type: jnp.ndarray      # (n_sources,) int32
    sersic_n: jnp.ndarray         # (n_sources,) float32 -- FROZEN-row n
    neighbor_idx: jnp.ndarray     # (n_sources, k_max) int32
    neighbor_mask: jnp.ndarray    # (n_sources, k_max) bool
    grid_x0: jnp.ndarray          # () float32 -- per-grid image origin
    grid_y0: jnp.ndarray          # () float32
    # Stage-2 free-n: True for TYPE=SERSIC rows whose Sersic index is FIT
    # (6th parameter u_n); False rows render from ``sersic_n`` above.
    free_n_mask: jnp.ndarray = None   # (n_sources,) bool


# ---------------------------------------------------------------------------
# Residual function (joint over all sources in the grid)
# ---------------------------------------------------------------------------


def make_residual_fn(
    *,
    data_stack: jnp.ndarray,        # (n_bands, H, W)
    invvar_stack: jnp.ndarray,      # (n_bands, H, W)
    pix_good_stack: jnp.ndarray,    # (n_bands, H, W) bool
    bg_basis_grid: jnp.ndarray,     # (H, W, M)  -- 2-D bg poly basis
    bg_coefs: jnp.ndarray,          # (n_bands, M)  -- frozen during pos step
    fluxes: jnp.ndarray,            # (n_sources, n_bands)  -- frozen
    psf_jaxes: list,                # n_bands PsfPolynomialJax (NOT batched)
    grid_x0: int,
    grid_y0: int,
    source_type: jnp.ndarray,       # (n_sources,) int32 (SOURCE_* codes)
    H_psf: int,
    W_psf: int,
    band_h: int,
    band_w: int,
    sersic_n: jnp.ndarray | None = None,   # (n_sources,) float32
    render_backend: str = "direct",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a JIT-friendly joint residual function.

    Input  ``p_flat``  shape ``(n_sources * 5,)``  = per-source
        ``(x, y, logre, ee1, ee2)`` in row-major order.
    Output ``r_flat``  shape ``(n_bands * band_h * band_w,)`` =
        ``sqrt(invvar) * (data - bg - sum_s flux_s * stamp_s)`` for
        good pixels, ``0`` for bad pixels.

    For STAR rows, the ``logre/ee1/ee2`` entries are ignored by the
    renderer (``render_source_stamp_at`` falls back to the raw PSF when
    ``source_type != SOURCE_EXP``); the gradient w.r.t. those entries
    is therefore zero, which is what we want.

    The returned callable also exposes:

    - ``residual_fn.core`` -- ``residual_core(p_flat, arrays)`` where
      ``arrays`` is a :class:`JointArrays` pytree of ALL per-grid data.
      The closure of ``residual_core`` holds only true statics (shapes,
      ``render_backend``, ``psf_jaxes``), so a step function jitted
      around it can be safely cached and reused across same-shape grids
      with the per-grid data flowing in as traced arguments.
    - ``residual_fn.arrays`` -- the :class:`JointArrays` built from this
      factory call's inputs (what ``residual_fn`` itself binds).
    """
    n_sources = int(source_type.shape[0])
    n_bands = int(data_stack.shape[0])
    if sersic_n is None:
        sersic_n = jnp.ones(n_sources, dtype=jnp.float32)

    # Pixel index meshes for the scatter step (static shapes only).
    py = jnp.arange(H_psf, dtype=jnp.int32)
    px = jnp.arange(W_psf, dtype=jnp.int32)

    def accumulate_stamps_one_band(
        stamps_b: jnp.ndarray,        # (n_sources, H_psf, W_psf)
        ix_top: jnp.ndarray,          # (n_sources,) int32
        iy_top: jnp.ndarray,          # (n_sources,) int32
    ) -> jnp.ndarray:                  # -> (band_h, band_w)
        # Per-source target pixel indices in the band image.
        iy_idx = iy_top[:, None, None] + py[None, :, None]   # (n_src, H_psf, 1)
        ix_idx = ix_top[:, None, None] + px[None, None, :]   # (n_src, 1, W_psf)
        iy_idx = jnp.broadcast_to(iy_idx, (n_sources, H_psf, W_psf))
        ix_idx = jnp.broadcast_to(ix_idx, (n_sources, H_psf, W_psf))
        valid = (
            (iy_idx >= 0) & (iy_idx < band_h) &
            (ix_idx >= 0) & (ix_idx < band_w)
        )
        iy_safe = jnp.clip(iy_idx, 0, band_h - 1)
        ix_safe = jnp.clip(ix_idx, 0, band_w - 1)
        vals = jnp.where(valid, stamps_b, 0.0).reshape(-1)
        return (
            jnp.zeros((band_h, band_w), dtype=stamps_b.dtype)
            .at[iy_safe.reshape(-1), ix_safe.reshape(-1)]
            .add(vals)
        )

    def residual_core(p_flat: jnp.ndarray, arrays: JointArrays) -> jnp.ndarray:
        # ALL per-grid data come in through ``arrays`` (traced), so this
        # function's closure holds only statics + the per-band PSFs.
        # The frozen background image / weight map are recomputed here;
        # they are cheap relative to the stamp rendering and XLA CSEs
        # them within one jitted step.
        sqrt_w = jnp.sqrt(jnp.where(arrays.pix_good_stack, arrays.invvar_stack, 0.0))
        bg_image = jnp.einsum("hwm,bm->bhw", arrays.bg_basis_grid, arrays.bg_coefs)
        data_minus_bg = arrays.data_stack - bg_image

        # Extended (EXP/DEV/SERSIC) rows keep their true shape values through
        # the NaN-guard substitution below; only STARs get inert placeholders.
        is_ext = (arrays.source_type != SOURCE_STAR)

        p = p_flat.reshape(n_sources, 5)
        pos_x = p[:, 0]
        pos_y = p[:, 1]
        logre_raw = p[:, 2]
        ee1_raw = p[:, 3]
        ee2_raw = p[:, 4]

        # JAX double-where pitfall guard: render_exp_profile_stamp_jax
        # contains arctan2(ee2, ee1) and other ops whose gradient is
        # NaN at degenerate inputs (e.g., ee1=ee2=0).  STAR catalog rows
        # have logre=ee1=ee2=0 by convention, so even though the
        # renderer's WHERE selects raw_psf for STAR rows (the EXP branch
        # output is unused), reverse-mode AD still differentiates the
        # masked-off branch and propagates NaN to the gradient w.r.t.
        # ee1/ee2 of the STAR rows.  Substitute a SAFE shape vector
        # (well away from degeneracy) for STAR rows; the output is
        # already discarded by the WHERE inside the renderer, so this
        # changes nothing forward-pass and zeros the (false) gradient
        # contribution.
        logre = jnp.where(is_ext, logre_raw, jnp.float32(0.0))
        ee1 = jnp.where(is_ext, ee1_raw, jnp.float32(1.0))
        ee2 = jnp.where(is_ext, ee2_raw, jnp.float32(0.0))

        # Integer top-left; round() has zero grad which is fine because
        # the sub-pixel shift lives inside lanczos_shift_image (which is
        # differentiable).
        ix_top = jnp.round(pos_x - arrays.grid_x0).astype(jnp.int32) - W_psf // 2
        iy_top = jnp.round(pos_y - arrays.grid_y0).astype(jnp.int32) - H_psf // 2

        residual_bands = []
        for b in range(n_bands):
            stamps = render_source_stamps(
                psf_jaxes[b], pos_x, pos_y, arrays.grid_x0, arrays.grid_y0,
                arrays.source_type, logre, ee1, ee2,
                sersic_n=arrays.sersic_n, render_backend=render_backend,
            )                                              # (n_sources, H_psf, W_psf)
            stamps_scaled = stamps * arrays.fluxes[:, b][:, None, None]
            model_b = accumulate_stamps_one_band(stamps_scaled, ix_top, iy_top)
            r_b = sqrt_w[b] * (data_minus_bg[b] - model_b)
            residual_bands.append(r_b)
        return jnp.stack(residual_bands, axis=0).reshape(-1)

    arrays = JointArrays(
        data_stack=data_stack,
        invvar_stack=invvar_stack,
        pix_good_stack=pix_good_stack,
        bg_basis_grid=bg_basis_grid,
        bg_coefs=bg_coefs,
        fluxes=fluxes,
        source_type=source_type,
        sersic_n=sersic_n,
        grid_x0=jnp.float32(grid_x0),
        grid_y0=jnp.float32(grid_y0),
    )

    def residual_fn(p_flat: jnp.ndarray) -> jnp.ndarray:
        return residual_core(p_flat, arrays)

    residual_fn.core = residual_core
    residual_fn.arrays = arrays
    return residual_fn


# ---------------------------------------------------------------------------
# J*v and J^T*v primitives via JAX autodiff
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Variable-projection residual function:
#   r(X) = sqrt(invvar) * (data - bg(B*(X)) - sum_s F*_s(X) * stamp_s(X))
# where F*(X), B*(X) are obtained at every call by the closed-form
# linear forced-phot solve.  This is mathematically equivalent to a
# full joint LSQR over (F, X, B) but only exposes X to the outer
# nonlinear solver (Schur complement / Golub-Pereyra "varpro").
# ---------------------------------------------------------------------------


def make_varpro_residual_fn(
    *,
    data_stack: jnp.ndarray,        # (n_bands, H, W)
    invvar_stack: jnp.ndarray,      # (n_bands, H, W)
    pix_good_stack: jnp.ndarray,    # (n_bands, H, W) bool
    bg_basis_grid: jnp.ndarray,     # (H, W, M)
    psf_jaxes: list,                # n_bands PsfPolynomialJax
    grid_x0: int,
    grid_y0: int,
    source_type: jnp.ndarray,       # (n_sources,) int32
    neighbor_idx: jnp.ndarray,      # (n_sources, k_max) int32
    neighbor_mask: jnp.ndarray,     # (n_sources, k_max) bool
    H_psf: int,
    W_psf: int,
    band_h: int,
    band_w: int,
    sersic_n: jnp.ndarray | None = None,   # (n_sources,) float32
    render_backend: str = "direct",
    bg_poly_order: int = 2,
    reg: float = 1e-9,
    free_n_mask: jnp.ndarray | None = None,   # (n_sources,) bool
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a JIT-friendly variable-projection residual function.

    Input  ``p_flat`` shape ``(n_sources * 6,)`` = per-source
        ``(x, y, logre, ee1, ee2, u_n)`` (logre/ee1/ee2 unused for STARs;
        ``u_n`` -- the logit-bijected Sersic index, see
        ``source_model.sersic_n_to_u`` -- is used ONLY for rows with
        ``free_n_mask`` True; all other rows render from the frozen
        ``sersic_n`` array).
    Output ``r_flat`` shape ``(n_bands * band_h * band_w,)``.

    For every call, we run the closed-form linear flux+bg solve PER
    BAND at the current positions/shapes, getting (F*, B*) per band,
    and form the residual w.r.t. that optimal flux+bg.  JAX autodiff
    through ``jnp.linalg.solve`` handles the chain rule of F*/B* w.r.t.
    X for free, so jvp/vjp of this residual are exactly the joint
    Jacobian-vector products of the full (F, X, B) system.

    The returned callable also exposes:

    - ``residual_fn.core`` -- ``residual_core(p_flat, arrays)`` where
      ``arrays`` is a :class:`VarproArrays` pytree of ALL per-grid data.
      The closure of ``residual_core`` holds only true statics (shapes,
      ``render_backend``, ``reg``, ``psf_jaxes``), so a step function
      jitted around it can be safely cached and reused across
      same-shape grids with the per-grid data flowing in as traced
      arguments.
    - ``residual_fn.arrays`` -- the :class:`VarproArrays` built from
      this factory call's inputs (what ``residual_fn`` itself binds).
    """
    n_sources = int(source_type.shape[0])
    n_bands = int(data_stack.shape[0])
    if sersic_n is None:
        sersic_n = jnp.ones(n_sources, dtype=jnp.float32)
    if free_n_mask is None:
        free_n_mask = jnp.zeros(n_sources, dtype=bool)
    M = n_terms_for_order(bg_poly_order)
    # ``reg`` is a run-level config constant (identical for every grid),
    # so it may stay baked into the traced program as a static.
    reg_jax = jnp.float32(reg)

    # Build the inner solve_band closure ONCE; it's pure JAX so
    # autograd traces through.  We use the un-jit version because we
    # want it inlined into the outer JIT trace.  Its closure holds only
    # static shapes; all data are call arguments.
    solver_inner_unjit, _solver_inner_jit = _make_solve_band_jit(
        H_psf=H_psf, W_psf=W_psf,
        band_h=band_h, band_w=band_w,
        n_sources=n_sources, k_max=int(neighbor_idx.shape[1]),
        M=M,
    )

    def accumulate_stamps_one_band(
        stamps_b: jnp.ndarray,        # (n_sources, H_psf, W_psf)
        ix_top: jnp.ndarray,          # (n_sources,) int32
        iy_top: jnp.ndarray,          # (n_sources,) int32
    ) -> jnp.ndarray:                  # (band_h, band_w)
        py = jnp.arange(H_psf, dtype=jnp.int32)
        px = jnp.arange(W_psf, dtype=jnp.int32)
        iy_idx = iy_top[:, None, None] + py[None, :, None]
        ix_idx = ix_top[:, None, None] + px[None, None, :]
        iy_idx = jnp.broadcast_to(iy_idx, (n_sources, H_psf, W_psf))
        ix_idx = jnp.broadcast_to(ix_idx, (n_sources, H_psf, W_psf))
        valid = (
            (iy_idx >= 0) & (iy_idx < band_h) &
            (ix_idx >= 0) & (ix_idx < band_w)
        )
        iy_safe = jnp.clip(iy_idx, 0, band_h - 1)
        ix_safe = jnp.clip(ix_idx, 0, band_w - 1)
        vals = jnp.where(valid, stamps_b, 0.0).reshape(-1)
        return (
            jnp.zeros((band_h, band_w), dtype=stamps_b.dtype)
            .at[iy_safe.reshape(-1), ix_safe.reshape(-1)]
            .add(vals)
        )

    def residual_core(p_flat: jnp.ndarray, arrays: VarproArrays) -> jnp.ndarray:
        # ALL per-grid data come in through ``arrays`` (traced), so this
        # function's closure holds only statics + the per-band PSFs.
        sqrt_w_stack = jnp.sqrt(jnp.where(arrays.pix_good_stack, arrays.invvar_stack, 0.0))
        bg_basis_flat = arrays.bg_basis_grid.reshape(-1, M)

        # Extended (EXP/DEV/SERSIC) rows keep their true shape values through
        # the NaN-guard substitution below; only STARs get inert placeholders.
        is_ext = (arrays.source_type != SOURCE_STAR)
        free_n = arrays.free_n_mask

        p = p_flat.reshape(n_sources, 6)
        pos_x = p[:, 0]
        pos_y = p[:, 1]
        logre = jnp.where(is_ext, p[:, 2], jnp.float32(0.0))
        ee1 = jnp.where(is_ext, p[:, 3], jnp.float32(1.0))
        ee2 = jnp.where(is_ext, p[:, 4], jnp.float32(0.0))
        # Stage-2 free-n: 6th parameter is the logit-bijected Sersic index.
        # Same NaN-guard idiom as logre/ee above: frozen rows get an inert
        # u=0 substituted (their output is discarded by the WHERE), and they
        # render from the frozen arrays.sersic_n exactly as before.
        u_n = jnp.where(free_n, p[:, 5], jnp.float32(0.0))
        n_render = jnp.where(free_n, sersic_u_to_n(u_n), arrays.sersic_n)

        ix_top, iy_top = stamp_top_left(
            pos_x, pos_y, arrays.grid_x0, arrays.grid_y0, H_psf, W_psf,
        )

        residual_bands = []
        for b in range(n_bands):
            stamps = render_source_stamps(
                psf_jaxes[b], pos_x, pos_y, arrays.grid_x0, arrays.grid_y0,
                arrays.source_type, logre, ee1, ee2,
                sersic_n=n_render, render_backend=render_backend,
            )                                              # (n_sources, H_psf, W_psf)
            # Closed-form linear solve for (F_b, B_b) given the current
            # stamps + data.  Differentiable through.
            F_b, B_b, _Fvar, _Bvar = solver_inner_unjit(
                arrays.data_stack[b], arrays.invvar_stack[b], arrays.pix_good_stack[b],
                stamps, ix_top, iy_top,
                arrays.neighbor_idx, arrays.neighbor_mask,
                bg_basis_flat, reg_jax,
            )
            # Build the model image: bg(B_b) + sum_s F_b[s] * stamp_s.
            stamps_scaled = stamps * F_b[:, None, None]
            sources_image = accumulate_stamps_one_band(stamps_scaled, ix_top, iy_top)
            bg_image = arrays.bg_basis_grid @ B_b   # (H, W)
            model_b = sources_image + bg_image
            r_b = sqrt_w_stack[b] * (arrays.data_stack[b] - model_b)
            residual_bands.append(r_b)
        return jnp.stack(residual_bands, axis=0).reshape(-1)

    arrays = VarproArrays(
        data_stack=data_stack,
        invvar_stack=invvar_stack,
        pix_good_stack=pix_good_stack,
        bg_basis_grid=bg_basis_grid,
        source_type=source_type,
        sersic_n=sersic_n,
        neighbor_idx=neighbor_idx,
        neighbor_mask=neighbor_mask,
        grid_x0=jnp.float32(grid_x0),
        grid_y0=jnp.float32(grid_y0),
        free_n_mask=free_n_mask,
    )

    def residual_fn(p_flat: jnp.ndarray) -> jnp.ndarray:
        return residual_core(p_flat, arrays)

    residual_fn.core = residual_core
    residual_fn.arrays = arrays
    return residual_fn


def Jv(residual_fn: Callable, p: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Jacobian-vector product J @ v (one forward-mode AD pass)."""
    _, out = jax.jvp(residual_fn, (p,), (v,))
    return out


def JTv(residual_fn: Callable, p: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
    """Vector-Jacobian product J^T @ u (one reverse-mode AD pass)."""
    _, vjp_fn = jax.vjp(residual_fn, p)
    (out,) = vjp_fn(u)
    return out


def chi2(residual_fn: Callable, p: jnp.ndarray) -> jnp.ndarray:
    """0.5 * ||r(p)||^2."""
    r = residual_fn(p)
    return 0.5 * jnp.sum(r * r)


def cheap_diagonal_preconditioner(
    *,
    fluxes: jnp.ndarray,             # (n_sources, n_bands) -- post-S1 fluxes
    invvar_stack: jnp.ndarray,       # (n_bands, H, W)
    pix_good_stack: jnp.ndarray,     # (n_bands, H, W)
    psf_jaxes: list,
    pos_x: jnp.ndarray,              # (n_sources,)
    pos_y: jnp.ndarray,
    grid_x0: int, grid_y0: int,
    source_type: jnp.ndarray,        # (n_sources,)
    logre: jnp.ndarray, ee1: jnp.ndarray, ee2: jnp.ndarray,
    H_psf: int, W_psf: int,
    band_h: int, band_w: int,
    sersic_n: jnp.ndarray | None = None,
    render_backend: str = "direct",
    fit_n: bool = False,
    free_n_mask: jnp.ndarray | None = None,   # (n_sources,) bool
    u_n: jnp.ndarray | None = None,           # (n_sources,) float32 (logit n)
) -> jnp.ndarray:
    """Cheap per-source-local approximation to ``diag(J^T J)``.

    With ``fit_n=False`` (default) the parameter vector per source is
    ``{x, y, logre, ee1, ee2}`` (5 entries); with ``fit_n=True`` a 6th
    entry for the logit-bijected Sersic index ``u_n`` is appended
    (sensitivity is exactly 0 for rows with ``free_n_mask`` False, whose
    render ignores ``u_n``).

    For each source s and parameter ``p_i``,
    we compute the local sensitivity::

        diag_local[s, i] = sum_{bands b} sum_{stamp pixels p}
            (flux_s_b * d stamp_s/d p_i)^2 * invvar(p) * pix_good(p)

    This is the EXACT diagonal entry ``(J^T J)_ii`` ONE WOULD GET IF:
      (a) fluxes were frozen at their input values (i.e., we're not
          including the variable-projection chain rule term, which only
          SHRINKS the diagonal -- so this is an upper bound on the true
          var-proj diag), and
      (b) there were no source-source overlap.

    Both approximations are fine for a preconditioner -- we just want
    each parameter scaled to roughly its curvature.  Cost is one
    autograd-of-render call per source (vmapped), which is dramatically
    cheaper than the full ``diagonal_JTJ`` scan over ``n_params``
    basis vectors.
    """
    from .source_model import render_source_stamp_at, SOURCE_EXP

    n_sources = int(source_type.shape[0])
    n_bands = len(psf_jaxes)
    K = 6 if fit_n else 5
    if sersic_n is None:
        sersic_n = jnp.ones(n_sources, dtype=jnp.float32)
    if free_n_mask is None:
        free_n_mask = jnp.zeros(n_sources, dtype=bool)
    if u_n is None:
        u_n = jnp.zeros(n_sources, dtype=jnp.float32)
    # Extended (EXP/DEV/SERSIC) rows keep their true shape values through
    # the NaN-guard substitution below; only STARs get inert placeholders.
    is_ext = (source_type != SOURCE_STAR)

    # Substitute SAFE shape values for STAR rows so render_exp_profile_stamp_jax's
    # arctan2(0, 0) doesn't NaN the gradient (same trick as in residual_fn).
    logre_safe = jnp.where(is_ext, logre, jnp.float32(0.0))
    ee1_safe = jnp.where(is_ext, ee1, jnp.float32(1.0))
    ee2_safe = jnp.where(is_ext, ee2, jnp.float32(0.0))
    u_safe = jnp.where(free_n_mask, u_n, jnp.float32(0.0))

    py = jnp.arange(H_psf, dtype=jnp.int32)
    px = jnp.arange(W_psf, dtype=jnp.int32)

    def per_source_diag(s_idx, x_s, y_s, lr_s, e1_s, e2_s, st_s, sn_s,
                        free_s, u_s):
        """Compute (J^T J)_ii diagonal for one source's K params."""
        # Place stamp:
        ix_top = jnp.round(x_s - grid_x0).astype(jnp.int32) - W_psf // 2
        iy_top = jnp.round(y_s - grid_y0).astype(jnp.int32) - H_psf // 2
        iy_idx = iy_top + py
        ix_idx = ix_top + px
        iy_safe = jnp.clip(iy_idx, 0, band_h - 1)
        ix_safe = jnp.clip(ix_idx, 0, band_w - 1)
        in_y = (iy_idx >= 0) & (iy_idx < band_h)
        in_x = (ix_idx >= 0) & (ix_idx < band_w)
        in_bounds = in_y[:, None] & in_x[None, :]   # (H_psf, W_psf)

        # Sum squared sensitivities across bands.
        diag_K = jnp.zeros(K, dtype=jnp.float32)
        for b in range(n_bands):
            invvar_b = invvar_stack[b]
            pixg_b = pix_good_stack[b]
            w_local = jnp.where(
                in_bounds & pixg_b[iy_safe[:, None], ix_safe[None, :]],
                invvar_b[iy_safe[:, None], ix_safe[None, :]],
                0.0,
            )                                          # (H_psf, W_psf)
            f_b = fluxes[s_idx, b]                     # scalar

            def stamp_fn(paramsK):
                xs = paramsK[0]; ys = paramsK[1]
                lr = paramsK[2]; e1 = paramsK[3]; e2 = paramsK[4]
                if fit_n:
                    # Free rows differentiate through the logit bijection;
                    # frozen rows render from their fixed sersic_n (the
                    # WHERE zeroes the u_n sensitivity for them).
                    n_eff = jnp.where(free_s, sersic_u_to_n(paramsK[5]), sn_s)
                else:
                    n_eff = sn_s
                return render_source_stamp_at(
                    psf_jaxes[b], xs, ys, grid_x0, grid_y0,
                    st_s, lr, e1, e2,
                    sersic_n=n_eff, render_backend=render_backend,
                ) * f_b

            params_list = [x_s, y_s, lr_s, e1_s, e2_s]
            if fit_n:
                params_list.append(u_s)
            paramsK = jnp.array(params_list, dtype=jnp.float32)

            # For each parameter, compute squared L2 of the partial derivative
            # over the (weighted) stamp pixels.
            def dirsq(unit_vec):
                _, deriv = jax.jvp(stamp_fn, (paramsK,), (unit_vec,))
                return jnp.sum((deriv ** 2) * w_local)

            for i in range(K):
                e_i = jnp.zeros(K, dtype=jnp.float32).at[i].set(1.0)
                diag_K = diag_K.at[i].add(dirsq(e_i))

        return diag_K

    diag = jax.vmap(per_source_diag, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
        jnp.arange(n_sources),
        pos_x, pos_y, logre_safe, ee1_safe, ee2_safe, source_type, sersic_n,
        free_n_mask, u_safe,
    )
    # Returns (n_sources, K).  Flatten in row-major to match p_flat layout.
    return diag.reshape(-1)


def diagonal_JTJ(
    residual_fn: Callable,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Compute ``diag(J^T J)`` at ``p`` -- one entry per parameter.

    Uses the identity ``[J^T J]_ii = ||J e_i||^2`` via a scan over the
    standard basis vectors.  This is exactly the pattern used by the
    legacy_gpu variance kernel (`legacy_gpu/lm.py::_make_variance_jit`).
    Cost: ``n_params`` forward-mode jvp calls.

    The diagonal is later used as the block-Jacobi preconditioner for
    CG: ``M^{-1} = diag(1 / (diag_JTJ + damp))`` makes the linear
    system close to an identity for the eigenmodes that don't involve
    inter-source coupling, dropping CG's iteration count from O(50-100)
    to O(5-20) for typical problems.
    """
    n_params = p.shape[0]
    eye = jnp.eye(n_params, dtype=p.dtype)

    def step(carry, e_i):
        _, Jei = jax.jvp(residual_fn, (p,), (e_i,))
        return carry, jnp.sum(Jei * Jei)

    _, diag = jax.lax.scan(step, None, eye)
    return diag


def make_joint_position_step_fn(
    residual_fn: Callable,
    n_sources: int,
    *,
    pos_step_max_pix: float = 0.5,
    max_shape_step: float = 0.15,
    cg_max_iters: int = 20,
    cg_tol: float = 1e-6,
    use_preconditioner: bool = True,
    n_params_per_source: int = 5,
) -> Callable:
    """Build a JIT-able joint LM step closure for a fixed grid shape.

    The returned function accepts dynamic ``p_flat``, bounds, ``damp``,
    (optional) per-source diagonal Hessian estimate, and an optional
    trailing ``arrays`` pytree; everything else (CG iterations, alpha
    grid size, n_sources, residual_fn) is static at trace time so the
    entire step compiles to a single XLA program.

    Two calling conventions for ``residual_fn``:

    - ``arrays=None`` (legacy): ``residual_fn(p_flat)`` -- a closure that
      has the grid data bound. Safe ONLY if the jitted step is NOT
      reused for a different grid (jit bakes the closed-over data in as
      compile-time constants).
    - ``arrays`` given: ``residual_fn`` must be a residual CORE taking
      ``(p_flat, arrays)`` (see :func:`make_residual_fn` /
      :func:`make_varpro_residual_fn` ``.core``). All per-grid data flow
      through the traced ``arrays`` argument, so the jitted step may be
      cached and reused across grids with the same shape signature.

    ``n_params_per_source`` selects the per-source parameter layout:

    - ``5`` (legacy / joint_lsqr): ``(x, y, logre, ee1, ee2)``. Parameter
      freedom: STAR rows move in (x, y) only; EXP rows additionally in
      shape (DEV/SERSIC shapes stay frozen, matching the historical
      joint_lsqr behavior).
    - ``6`` (varpro Stage-2): ``(x, y, logre, ee1, ee2, u_n)``. Freedom:
      STAR (x, y); any extended row (EXP/DEV/SERSIC) also shape; rows
      with ``arrays.free_n_mask`` True (SERSIC passing the S/N gate)
      also ``u_n``. Requires ``arrays`` (the free-n mask rides in the
      pytree) and the ``un_lower/un_upper`` bound arguments.

    The caller is responsible for wrapping this with ``jax.jit`` and
    caching the result keyed on the grid's shape signature
    (n_sources, n_bands, band_h, band_w, H_psf, W_psf).
    """
    K = int(n_params_per_source)
    if K not in (5, 6):
        raise ValueError(f"n_params_per_source must be 5 or 6; got {K}")
    pos_clip = jnp.float32(pos_step_max_pix)
    shape_clip = jnp.float32(max_shape_step)
    clip_per_src = jnp.array(
        [pos_clip, pos_clip, shape_clip, shape_clip, shape_clip]
        + ([shape_clip] if K == 6 else []),   # u_n moves on the shape scale
        dtype=jnp.float32,
    )
    clip_flat = jnp.tile(clip_per_src, n_sources)
    alphas = jnp.asarray(2.0 ** np.arange(-10, 1), dtype=jnp.float32)
    cg_max_iters_int = int(cg_max_iters)
    cg_tol_f = float(cg_tol)
    use_pre = bool(use_preconditioner)

    def step(
        p_flat,
        source_type,
        pos_lower_x, pos_upper_x,
        pos_lower_y, pos_upper_y,
        logre_lower, logre_upper,
        ee1_lower, ee1_upper,
        ee2_lower, ee2_upper,
        damp,
        diag_JTJ,                # (n_params,) -- only used if use_pre is True
        arrays=None,             # per-grid data pytree (traced); see docstring
        un_lower=None,           # (n_sources,) -- required when K == 6
        un_upper=None,
    ):
        # Bind the per-grid data to the residual core. Binding via a
        # lambda here is SAFE: ``arrays`` is a traced ARGUMENT of this
        # (jitted) step, not a Python constant, so the compiled program
        # stays valid for any same-shape grid's data.
        if arrays is None:
            rfn = residual_fn
        else:
            rfn = lambda p: residual_fn(p, arrays)  # noqa: E731

        if K == 6:
            # Stage-2 layout. Per-row freedom: STAR -> (x, y); extended
            # (EXP/DEV/SERSIC) -> + (logre, ee1, ee2); free-n SERSIC
            # (arrays.free_n_mask) -> + u_n.
            if arrays is None:
                raise ValueError(
                    "n_params_per_source=6 requires the per-grid `arrays` "
                    "pytree (free_n_mask rides in it)")
            if un_lower is None or un_upper is None:
                raise ValueError(
                    "n_params_per_source=6 requires un_lower/un_upper bounds")
            ones_col = jnp.ones(n_sources, dtype=jnp.float32)
            ext_col = (source_type != SOURCE_STAR).astype(jnp.float32)
            free_col = arrays.free_n_mask.astype(jnp.float32)
            keep_mask = jnp.stack(
                [ones_col, ones_col, ext_col, ext_col, ext_col, free_col],
                axis=1,
            ).reshape(-1)
        else:
            # Legacy 5-param layout (joint_lsqr): EXP-only shape freedom.
            is_exp_flat = jnp.repeat(source_type == SOURCE_EXP, 5)
            shape_mask_per_src = jnp.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
            shape_mask_flat = jnp.tile(shape_mask_per_src, n_sources)
            keep_mask = jnp.where(is_exp_flat, jnp.float32(1.0), shape_mask_flat)

        r0 = rfn(p_flat)
        chi2_old = 0.5 * jnp.sum(r0 * r0)

        _, vjp_fn = jax.vjp(rfn, p_flat)
        (g,) = vjp_fn(r0)
        g = g * keep_mask

        def matvec(v):
            v_masked = v * keep_mask
            _, Jv_out = jax.jvp(rfn, (p_flat,), (v_masked,))
            (JTJv,) = vjp_fn(Jv_out)
            return (JTJv + damp * v_masked) * keep_mask

        if use_pre:
            d_safe = jnp.where(keep_mask > 0, jnp.maximum(diag_JTJ, 1e-30), 1.0)
            M_inv_diag = keep_mask / (d_safe + damp)

            def M_inv(v):
                return v * M_inv_diag * keep_mask

            delta_raw, _info = jax.scipy.sparse.linalg.cg(
                matvec, -g, M=M_inv, maxiter=cg_max_iters_int, tol=cg_tol_f,
            )
        else:
            delta_raw, _info = jax.scipy.sparse.linalg.cg(
                matvec, -g, maxiter=cg_max_iters_int, tol=cg_tol_f,
            )

        delta_raw = delta_raw * keep_mask
        delta_raw = jnp.where(jnp.isfinite(delta_raw), delta_raw, 0.0)
        delta = jnp.clip(delta_raw, -clip_flat, clip_flat)

        lower_cols = [pos_lower_x, pos_lower_y, logre_lower, ee1_lower, ee2_lower]
        upper_cols = [pos_upper_x, pos_upper_y, logre_upper, ee1_upper, ee2_upper]
        if K == 6:
            lower_cols.append(un_lower)
            upper_cols.append(un_upper)
        lower_per_src = jnp.stack(lower_cols, axis=1).reshape(-1)
        upper_per_src = jnp.stack(upper_cols, axis=1).reshape(-1)

        def clip_p(p):
            return jnp.clip(p, lower_per_src, upper_per_src)

        def alpha_chi2(alpha):
            p_trial = clip_p(p_flat + alpha * delta)
            r_trial = rfn(p_trial)
            return 0.5 * jnp.sum(r_trial * r_trial), p_trial

        chi2_arr, p_arr = jax.vmap(alpha_chi2)(alphas)
        accept_mask = chi2_arr < chi2_old
        masked_chi2 = jnp.where(accept_mask, chi2_arr, jnp.inf)
        best_idx = jnp.argmin(masked_chi2)
        any_accept = jnp.any(accept_mask)
        p_new = jnp.where(any_accept, p_arr[best_idx], clip_p(p_flat))
        chi2_new = jnp.where(any_accept, chi2_arr[best_idx], chi2_old)
        alpha_used = jnp.where(any_accept, alphas[best_idx], jnp.float32(0.0))
        delta_norm = jnp.linalg.norm(delta)

        return p_new, chi2_old, chi2_new, alpha_used, delta_norm

    return step


def joint_position_step(
    *,
    residual_fn: Callable,
    p_flat: jnp.ndarray,
    source_type: jnp.ndarray,
    pos_lower_x: jnp.ndarray, pos_upper_x: jnp.ndarray,
    pos_lower_y: jnp.ndarray, pos_upper_y: jnp.ndarray,
    logre_lower: jnp.ndarray, logre_upper: jnp.ndarray,
    ee1_lower: jnp.ndarray, ee1_upper: jnp.ndarray,
    ee2_lower: jnp.ndarray, ee2_upper: jnp.ndarray,
    pos_step_max_pix: float = 0.5,
    max_shape_step: float = 0.15,
    damp: float = 1e-3,
    cg_max_iters: int = 30,
    cg_tol: float = 1e-6,
    preconditioner_diag: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, dict]:
    """Take ONE joint LM step in flat 5*n_sources param space.

    Convenience wrapper around :func:`make_joint_position_step_fn`
    that builds (and discards) a non-cached step closure each call.
    For production use, call ``make_joint_position_step_fn`` once and
    JIT-cache the result keyed on the grid shape signature.
    """
    n_sources = int(source_type.shape[0])
    use_pre = preconditioner_diag is not None
    step_fn = make_joint_position_step_fn(
        residual_fn, n_sources,
        pos_step_max_pix=pos_step_max_pix,
        max_shape_step=max_shape_step,
        cg_max_iters=cg_max_iters,
        cg_tol=cg_tol,
        use_preconditioner=use_pre,
    )

    if use_pre:
        diag_arg = preconditioner_diag
    else:
        diag_arg = jnp.zeros_like(p_flat)

    p_new, chi2_old, chi2_new, alpha_used, delta_norm = step_fn(
        p_flat, source_type,
        pos_lower_x, pos_upper_x,
        pos_lower_y, pos_upper_y,
        logre_lower, logre_upper,
        ee1_lower, ee1_upper,
        ee2_lower, ee2_upper,
        jnp.float32(damp),
        diag_arg,
    )
    info = {
        "chi2_old": chi2_old,
        "chi2_new": chi2_new,
        "alpha_used": alpha_used,
        "delta_norm": delta_norm,
    }
    return p_new, info


__all__ = [
    "JointArrays",
    "VarproArrays",
    "make_residual_fn",
    "make_varpro_residual_fn",
    "Jv",
    "JTv",
    "chi2",
    "diagonal_JTJ",
    "cheap_diagonal_preconditioner",
    "joint_position_step",
    "make_joint_position_step_fn",
]
