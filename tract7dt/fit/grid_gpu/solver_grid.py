"""3-staged Levenberg-Marquardt solver for one image grid.

Multi-band (positions shared across bands; fluxes per (source, band);
bg coefs per band) joint fit of:

- ``n_sources`` PointSource positions (x, y) shared across all bands
- ``n_sources x n_bands`` per-source per-band fluxes
- ``n_bands x n_bg_coefs`` 2-D polynomial background

Mirrors the per-patch CPU mode's 3-stage thaw pattern but at GRID
SCALE via :mod:`.forced_phot_sparse`'s sparse-stamp accumulator:

- **S1**: positions FROZEN, fit (fluxes + bg coefs) per band. Linear
  problem - one closed-form Cholesky per band via
  ``solve_band_sparse``.
- **S2**: positions THAWED, freeze (fluxes + bg coefs). Per-source
  Newton step on (x_s, y_s); chi^2 sums over all bands.
- **S3**: all thawed. Alternate linear flux+bg solve (S1-style) +
  position Newton step (S2-style) until convergence.

Per-source position priors come from the input catalog's POS_ERR
column; positions are clipped to ``init +/- pos_max_shift_pix``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Callable, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

# Per-source position-step CHUNK size (DEFAULT ON). The position Newton step
# vmaps over all sources at once, materializing a (n_sources x n_bands x stamp^2)
# working set (x11 for the line search, x autodiff for the Hessian) -- the
# per-grid PEAK memory (~10 GiB for 1750 src x 20 bands). Chunking that vmap with
# lax.map so only CHUNK sources are live at once cuts the peak ~n_sources/CHUNK
# (10.9 -> ~4 GiB at 512), which enables safe multi-worker GPU packing on a
# 24 GB card.
#
# vmap vs lax.map are not bit-identical (fp non-associativity in XLA lowering,
# amplified through the iterative solve), BUT the change is WITHIN NOISE.
# Validated on T08285 (chunked vs single-vmap reference): max |dflux|/FLUXERR =
# 0.63 sigma, 0% of sources move >1 sigma, per-band ZP within ~5e-5 mag, and
# BRIGHT/calibration sources (mag<18) unchanged to median 0 / <0.2 mmag; the
# large *relative*/mmag diffs are only at the faint, noise-dominated end
# (mag>22, <0.04 sigma). DEFAULT ON for the multi-worker packing speedup
# (~1.8x -> ~2.15x on a 24GB x4 box).
#
# CAVEAT: with multi-worker packing (workers_per_gpu>1) the result is NOT
# bit-reproducible run-to-run -- concurrent GPU execution makes the flux-solve
# scatter-add nondeterministic at the fp level, so two runs of the same config
# differ, always WITHIN the above noise envelope (<1 sigma). Set
# TRACT7DT_POS_STEP_CHUNK=0 to restore the legacy single-vmap path (which, at
# 1 worker/GPU, is bit-exact and fully reproducible). 0 or >= n_sources off.
_POS_STEP_CHUNK = int(os.environ.get("TRACT7DT_POS_STEP_CHUNK", "512"))

# Chunking guards for the MIXED (extended-source) per-source Newton vmap
# in stage2_source_newton_step. Unlike the star path above (which chunks
# whenever n_sources > chunk), the mixed path is SIZE-triggered on the
# dominant working-set estimate -- the per-neighbour padded-window gather
# (n_sources x n_bands x k_local x H_pad x W_pad fp32):
#   - TRIGGER: chunk only when the full-vmap gather exceeds this many
#     bytes. Small fields (UDS extended: ~1 GB) keep the exact legacy
#     single-vmap program (bit-compat); crowded many-band fields
#     (T08285 + extended: ~96 GB estimate, observed 9-26 GiB single
#     allocations that OOM'd every grid) chunk to survive.
#   - BUDGET: the chunk size is chosen so the gather x the AD factor
#     stays under this many bytes (clamped to [16,
#     TRACT7DT_POS_STEP_CHUNK]). The per-source Newton's 5-param
#     Hessian (forward-over-reverse) carries the primal + 5 tangent
#     copies of the whole per-chunk working set -- measured exactly
#     6.0x on T08285 (a 60-source chunk with k_local~181 allocated
#     19.03 GiB = 6 x the single-copy gather) -- so the sizing
#     multiplies per-source bytes by _SRC_STEP_AD_COPIES. At T08285
#     scale that lands at the 16-source floor (~5 GiB peak, fits both
#     the packed 2-worker cap and the exclusive retry).
# The chunk-size CAP for the mixed path comes from its own env
# (TRACT7DT_SRC_STEP_CHUNK_CAP, exported by the pipeline from the
# user's TRACT7DT_POS_STEP_CHUNK BEFORE the wpg=1 zeroing below):
# the pipeline forces TRACT7DT_POS_STEP_CHUNK=0 for workers at
# 1 worker/GPU to keep the STAR path bit-exact, but the mixed-path
# chunking is a SURVIVAL requirement on crowded extended fields (the
# unchunked gather is tens of GB), not a packing optimization, so it
# must not be silently disabled by the wpg decision. A user-explicit
# TRACT7DT_POS_STEP_CHUNK=0 still disables both paths.
_SRC_STEP_CHUNK_TRIGGER_BYTES = 2 * 1024 ** 3
_SRC_STEP_CHUNK_BUDGET_BYTES = 3 * 1024 ** 3
_SRC_STEP_AD_COPIES = 6
_SRC_STEP_CHUNK_CAP = int(os.environ.get(
    "TRACT7DT_SRC_STEP_CHUNK_CAP",
    os.environ.get("TRACT7DT_POS_STEP_CHUNK", "512"),
))

from ..optimize.gpu.kernels.lanczos import lanczos_shift_image
from .bg_polynomial import N_TERMS_QUADRATIC, n_terms_for_order
from .forced_phot_sparse import (
    _has_exp_sources, collect_band_sparse, launch_band_sparse,
    DEFAULT_K_MAX, GridBandInputs, SparseForcedPhotResult,
    build_neighbor_list, solve_band_sparse,
)
from .psf_polynomial import (
    PsfPolynomialJax, _spatial_basis_jax, evaluate_psf_at,
)
from .source_model import (
    SOURCE_DEV, SOURCE_EXP, SOURCE_SERSIC, SOURCE_STAR,
    render_source_stamp_at, render_source_stamps,
    sersic_n_to_u, sersic_u_to_n,
)


# ---------------------------------------------------------------------------
# Multi-band solver inputs / outputs
# ---------------------------------------------------------------------------


@dataclass
class GridMultiBandInputs:
    """All per-band inputs for one grid + shared per-source position state.

    Per-band data are stored as a list of ``GridBandInputs``; all
    bands MUST share the same n_sources (positions are shared).
    """
    bands: list[GridBandInputs]
    bandnames: list[str]
    # Per-source priors + bounds (in FULL-image pixel coords).
    pos_init_x: np.ndarray   # (n_sources,) - input catalog positions
    pos_init_y: np.ndarray
    pos_err: np.ndarray      # (n_sources,) Gaussian prior sigma in pix
    pos_lower_x: np.ndarray  # (n_sources,) - hard bound
    pos_upper_x: np.ndarray
    pos_lower_y: np.ndarray
    pos_upper_y: np.ndarray
    source_type: np.ndarray | None = None
    logre_init: np.ndarray | None = None
    ee1_init: np.ndarray | None = None
    ee2_init: np.ndarray | None = None
    logre_lower: np.ndarray | None = None
    logre_upper: np.ndarray | None = None
    ee1_lower: np.ndarray | None = None
    ee1_upper: np.ndarray | None = None
    ee2_lower: np.ndarray | None = None
    ee2_upper: np.ndarray | None = None
    # Sersic index per source (used by DEV/SERSIC rendering; EXP renders
    # from its dedicated constants) + trace-time renderer choice.
    sersic_n: np.ndarray | None = None
    render_backend: str = "direct"
    # Phase-B stamp promotion: the bands' psf_jax may be zero-PADDED to a
    # larger tier (used by the LINEAR flux solves, where wing coverage sets
    # the flux accuracy). The position/shape Newton and joint steps use the
    # NATIVE stamps below (their per-source padded windows scale ~tier^2 in
    # GPU memory, and shape/position accuracy does not need the big box).
    psf_jaxes_native: list | None = None

    def newton_psf_jaxes(self):
        if self.psf_jaxes_native is not None:
            return self.psf_jaxes_native
        return [b.psf_jax for b in self.bands]

    @property
    def n_sources(self) -> int:
        return int(self.pos_init_x.shape[0])

    @property
    def n_bands(self) -> int:
        return len(self.bands)


@dataclass
class GridMultiBandResult:
    """Per-grid 3-stage LM result."""
    pos_x: np.ndarray                      # (n_sources,) final
    pos_y: np.ndarray                      # (n_sources,)
    logre: np.ndarray                      # (n_sources,) final exp shape
    ee1: np.ndarray
    ee2: np.ndarray
    fluxes: np.ndarray                     # (n_sources, n_bands)
    bg_coefs: np.ndarray                   # (n_bands, N_TERMS_QUADRATIC)
    flux_var: np.ndarray                   # (n_sources, n_bands)
    bg_var: np.ndarray                     # (n_bands, N_TERMS_QUADRATIC)
    pos_var: np.ndarray                    # (n_sources, 2) - x, y variances
    n_outer_iters: int                     # S3 alternating iters used
    s1_converged: bool
    s2_converged: bool
    s3_converged: bool
    # Per-source: True for sources whose position step was still above
    # tolerance when S3 stopped (only non-empty under the quantile stop,
    # pos_stop_unconverged_frac > 0). Their positions/fluxes are the last
    # iterate, not a converged optimum -- flag, don't trust astrometry.
    position_stranded: "np.ndarray | None" = None
    # Stage-2 free-n (varpro): final per-source Sersic index (fitted for
    # free-n SERSIC rows, seed value otherwise) and the freeze flag
    # (True where TYPE == SERSIC and n was NOT freed -- fit_sersic_n off,
    # non-varpro solver, or S/N below n_fit_min_snr).
    sersic_n_fit: "np.ndarray | None" = None
    sersic_n_frozen: "np.ndarray | None" = None
    # Per-source: True for a DEV/SERSIC (extended, non-STAR) row whose grid
    # ran a SHAPE-FREEZING solver (position_solver in {per_source,
    # joint_lsqr}). Those solvers thaw shape only for EXP, so a DEV/SERSIC
    # row's Re/ELL/THETA are echoed verbatim from the catalog seed -- the
    # reported shape was NOT fit (the flux is a real measurement conditioned
    # on the assumed shape, exactly like a frozen-n flux). False for EXP
    # (shape IS fit under per_source), for STAR (no shape), and under the
    # shape-fitting solvers (joint_varpro; island, which overrides this).
    shape_frozen: "np.ndarray | None" = None
    # 1-sigma uncertainty of the fitted Sersic index (free-n rows only;
    # NaN elsewhere). Diagonal / flux-marginalized -- see
    # :func:`_varpro_sersic_n_err`.
    sersic_n_err: "np.ndarray | None" = None
    # (n_bands,) blank-response offset b_hat that was subtracted from this
    # grid's STAR fluxes (flux units of `fluxes`); None when the correction
    # did not run. Output-only record (REGISTER R09).
    blank_response_bhat: "np.ndarray | None" = None


# ---------------------------------------------------------------------------
# Stage 1: linear flux + bg solve, per band, positions frozen.
# ---------------------------------------------------------------------------


def stage1_solve(
    multi_inputs: GridMultiBandInputs,
    pos_x: np.ndarray, pos_y: np.ndarray,
    logre: np.ndarray | None,
    ee1: np.ndarray | None,
    ee2: np.ndarray | None,
    neighbor_idx: np.ndarray, neighbor_mask: np.ndarray,
    reg: float = 1e-9,
    bg_poly_order: int = 2,
    return_stamps: bool = False,
    data_override: "list | None" = None,
) -> "tuple[list[SparseForcedPhotResult], bool] | tuple[list[SparseForcedPhotResult], bool, list | None]":
    """Run S1: sparse linear forced-phot solve.

    ``return_stamps=True`` additionally returns the per-band UNSCALED stamp
    DeviceArrays rendered for the solves (all-star path only; None for
    exp-source grids). The position step of the SAME
    iteration runs at the SAME positions, so it can reuse these buffers
    instead of re-rendering -- bit-identical by construction.

    ``data_override`` (wing peeling): optional list of per-band effective
    data images (data - wing) used in place of each band's own ``data``.
    The original band inputs are NEVER mutated -- shallow dataclass copies
    carry the effective data into the solves.

    Positions are frozen at ``(pos_x, pos_y)``.

    Per-band loop over :func:`launch_band_sparse` /
    :func:`collect_band_sparse` (JAX kernels launch async, so the loop
    pipelines well; lowest peak GPU memory).

    Returns a list of per-band ``SparseForcedPhotResult`` and a single
    bool indicating whether ALL bands converged.
    """
    pos_x_jax = jnp.asarray(pos_x)
    pos_y_jax = jnp.asarray(pos_y)
    n_sources = int(pos_x.shape[0])
    source_type_jax = jnp.asarray(
        multi_inputs.source_type
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=np.int32),
        dtype=jnp.int32,
    )
    logre_jax = jnp.asarray(
        logre
        if logre is not None
        else (
            multi_inputs.logre_init
            if multi_inputs.logre_init is not None
            else np.zeros(n_sources, dtype=np.float32)
        ),
        dtype=jnp.float32,
    )
    ee1_jax = jnp.asarray(
        ee1
        if ee1 is not None
        else (
            multi_inputs.ee1_init
            if multi_inputs.ee1_init is not None
            else np.zeros(n_sources, dtype=np.float32)
        ),
        dtype=jnp.float32,
    )
    ee2_jax = jnp.asarray(
        ee2
        if ee2 is not None
        else (
            multi_inputs.ee2_init
            if multi_inputs.ee2_init is not None
            else np.zeros(n_sources, dtype=np.float32)
        ),
        dtype=jnp.float32,
    )
    sersic_n_jax = jnp.asarray(
        multi_inputs.sersic_n
        if multi_inputs.sersic_n is not None
        else np.ones(n_sources, dtype=np.float32),
        dtype=jnp.float32,
    )
    for inp_b in multi_inputs.bands:
        inp_b.pos_x = pos_x_jax
        inp_b.pos_y = pos_y_jax
        inp_b.source_type = source_type_jax
        inp_b.logre = logre_jax
        inp_b.ee1 = ee1_jax
        inp_b.ee2 = ee2_jax
        inp_b.sersic_n = sersic_n_jax
        inp_b.render_backend = multi_inputs.render_backend

    # Wing peeling: solve against (data - wing) via shallow copies; the
    # caller's band inputs keep their pristine data arrays.
    bands_for_solve = multi_inputs.bands
    if data_override is not None:
        import dataclasses as _dc
        bands_for_solve = [
            _dc.replace(inp_b, data=jnp.asarray(d, dtype=jnp.float32))
            for inp_b, d in zip(multi_inputs.bands, data_override)
        ]

    stamps_list: "list | None" = None
    # Launch/collect with a bounded in-flight window: band b's host-side
    # render (~100 ms of eager-vmap dispatch, the dominant per-band
    # cost) overlaps the previous bands' GPU solves instead of draining
    # the pipeline with a per-band np.asarray sync (~0.9 s saved per
    # 20-band call, x ~235 calls/run). Same XLA programs, same inputs,
    # same single-stream submission order -> bit-identical results
    # (verified by output digest at production shapes). The window only
    # bounds in-flight buffers as belt-and-suspenders under MPS; the
    # loop is host-bound so the queue never actually deepens.
    # Per-grid constants are uploaded once, not per band.
    if return_stamps and not _has_exp_sources(bands_for_solve[0]):
        # Stamp reuse is only exact on the all-star path: the position
        # step renders pure-PSF stamps, which for exp grids differ
        # from the lin solve's galaxy-convolved stamps.
        stamps_list = []
    nbr_idx_jax = jnp.asarray(neighbor_idx)
    nbr_mask_jax = jnp.asarray(neighbor_mask)
    reg_jax = jnp.asarray(reg, dtype=jnp.float32)
    window = 8
    pending: list = []
    results = []
    for inp_b in bands_for_solve:
        p = launch_band_sparse(
            inp_b, neighbor_idx, neighbor_mask, reg=reg,
            bg_poly_order=bg_poly_order,
            nbr_idx_jax=nbr_idx_jax, nbr_mask_jax=nbr_mask_jax,
            reg_jax=reg_jax,
        )
        if stamps_list is not None:
            stamps_list.append(p.stamps)
        pending.append(p)
        if len(pending) > window:
            results.append(collect_band_sparse(pending.pop(0)))
    while pending:
        results.append(collect_band_sparse(pending.pop(0)))
    all_ok = all(r.converged for r in results)
    if return_stamps:
        return results, all_ok, stamps_list
    return results, all_ok


_LOCAL_POSITION_STEP_CACHE: dict[tuple[int, ...], "jax.stages.Wrapped"] = {}


def _make_local_position_step_jit(
    n_bands: int,
    band_h: int,
    band_w: int,
    H_psf: int,
    W_psf: int,
    H_pad: int,
    W_pad: int,
    M: int,
    n_terms: int,
    k_local: int,
):
    """Build a fused polynomial-PSF local position-step kernel.

    This fast path is mathematically the same as the generic local residual
    path, but stacks per-band polynomial PSFs so XLA can compile the whole
    source/band update instead of launching many small eager kernels.
    """

    ky_pad = jnp.arange(H_pad) - ((H_pad - H_psf) // 2)
    kx_pad = jnp.arange(W_pad) - ((W_pad - W_psf) // 2)
    py_pad = jnp.arange(H_pad)[:, None]
    px_pad = jnp.arange(W_pad)[None, :]
    eye2 = jnp.eye(2, dtype=jnp.float32)

    def eval_psf_stack(psf_coefs_stack, x_origins, x_scales, y_origins, y_scales, x, y):
        X = (x - x_origins) / x_scales
        Y = (y - y_origins) / y_scales
        basis = _spatial_basis_jax(X, Y, n_terms=n_terms)  # (n_bands, n_terms)
        return jnp.einsum("bhwt,bt->bhw", psf_coefs_stack, basis)

    def per_source_step(
        s_idx, x, y, fluxes_s, damp_s, mux, muy, prior_inv_sigma,
        ix_top_s, iy_top_s, nbr_idx_row, nbr_mask_row,
        data_stack, invvar_stack, pix_good_stack, bg_basis_grid, bg_coefs,
        stamps_stack, ix_top, iy_top,
        psf_coefs_stack, x_origins, x_scales, y_origins, y_scales,
        grid_x0_j, grid_y0_j, max_step_pix_j,
        damp_min_j, damp_max_j, damp_up_j, damp_down_j,
    ):
        ys_pad = iy_top_s + ky_pad[:, None]
        xs_pad = ix_top_s + kx_pad[None, :]
        in_bounds_pad = (
            (ys_pad >= 0) & (ys_pad < band_h) &
            (xs_pad >= 0) & (xs_pad < band_w)
        )
        safe_y_pad = jnp.clip(ys_pad, 0, band_h - 1)
        safe_x_pad = jnp.clip(xs_pad, 0, band_w - 1)

        data_pad = data_stack[:, safe_y_pad, safe_x_pad]
        invvar_pad = invvar_stack[:, safe_y_pad, safe_x_pad]
        pixg_pad = pix_good_stack[:, safe_y_pad, safe_x_pad]
        bg_under = bg_basis_grid[safe_y_pad, safe_x_pad, :]
        bg_pad = jnp.einsum("hwm,bm->bhw", bg_under, bg_coefs)

        def neighbor_pad(t_idx, mask):
            stamp_t = stamps_stack[:, t_idx]
            off_y = iy_top[t_idx] - iy_top_s + ((H_pad - H_psf) // 2)
            off_x = ix_top[t_idx] - ix_top_s + ((W_pad - W_psf) // 2)
            sy = py_pad - off_y
            sx = px_pad - off_x
            in_stamp = (sy >= 0) & (sy < H_psf) & (sx >= 0) & (sx < W_psf)
            safe_sy = jnp.clip(sy, 0, H_psf - 1)
            safe_sx = jnp.clip(sx, 0, W_psf - 1)
            vals = stamp_t[:, safe_sy, safe_sx]
            return jnp.where(mask, jnp.where(in_stamp[None, :, :], vals, 0.0), 0.0)

        other_pad = jnp.sum(jax.vmap(neighbor_pad)(nbr_idx_row, nbr_mask_row), axis=0)
        dmo_pad = jnp.where(in_bounds_pad[None, :, :], data_pad - bg_pad - other_pad, 0.0)
        w_pad = jnp.where(
            in_bounds_pad[None, :, :],
            jnp.where(pixg_pad, invvar_pad, 0.0),
            0.0,
        )

        iy_orig = iy_top_s + (H_psf // 2)
        ix_orig = ix_top_s + (W_psf // 2)

        def chi2_xy(p):
            x_t, y_t = p[0], p[1]
            x_grid = x_t - grid_x0_j
            y_grid = y_t - grid_y0_j
            ix_t = jax.lax.stop_gradient(jnp.round(x_grid).astype(jnp.int32))
            iy_t = jax.lax.stop_gradient(jnp.round(y_grid).astype(jnp.int32))
            dx = x_grid - ix_t.astype(jnp.float32)
            dy = y_grid - iy_t.astype(jnp.float32)

            start_h = (iy_t - iy_orig + ((H_pad - H_psf) // 2)).astype(jnp.int32)
            start_w = (ix_t - ix_orig + ((W_pad - W_psf) // 2)).astype(jnp.int32)
            zero_i32 = jnp.int32(0)
            dmo_under = jax.lax.dynamic_slice(
                dmo_pad, (zero_i32, start_h, start_w),
                (n_bands, H_psf, W_psf),
            )
            w_under = jax.lax.dynamic_slice(
                w_pad, (zero_i32, start_h, start_w),
                (n_bands, H_psf, W_psf),
            )

            raw_stack = eval_psf_stack(
                psf_coefs_stack, x_origins, x_scales, y_origins, y_scales, x_t, y_t,
            )
            shifted_stack = jax.vmap(
                lambda s: lanczos_shift_image(s, dx, dy),
            )(raw_stack)
            stamp_at_xy = shifted_stack * fluxes_s[:, None, None]
            diff = dmo_under - stamp_at_xy
            chi2 = jnp.sum(diff * diff * w_under)
            chi2 = chi2 + (prior_inv_sigma ** 2) * (
                (x_t - mux) ** 2 + (y_t - muy) ** 2
            )
            return chi2

        p0 = jnp.array([x, y], dtype=jnp.float32)
        chi2_old, g = jax.value_and_grad(chi2_xy)(p0)
        H = jax.hessian(chi2_xy)(p0)
        diag_H = jnp.diag(H)
        diag_safe = jnp.where(jnp.abs(diag_H) > 1e-30, jnp.abs(diag_H), 1e-30)
        D = jnp.float32(1.0) / jnp.sqrt(diag_safe)
        H_scaled = H * D[:, None] * D[None, :]
        g_scaled = g * D
        H_damped = H_scaled + damp_s * eye2
        H64 = H_damped.astype(jnp.float64)
        g64 = g_scaled.astype(jnp.float64)
        delta_scaled = jnp.linalg.solve(H64, -g64).astype(jnp.float32)
        delta = D * delta_scaled
        delta = jnp.clip(delta, -max_step_pix_j, max_step_pix_j)
        delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))

        # 11-alpha log-spaced line search (mirror Tractor's tryUpdates,
        # capped at alpha=1.0 since delta is already clipped).
        alphas = jnp.asarray(2.0 ** np.arange(-10, 1), dtype=jnp.float32)
        chi2_arr, p_arr = jax.vmap(lambda a: (chi2_xy(p0 + a * delta), p0 + a * delta))(alphas)
        accept_mask = chi2_arr < chi2_old
        masked_chi2 = jnp.where(accept_mask, chi2_arr, jnp.inf)
        best_idx = jnp.argmin(masked_chi2)
        any_accept = jnp.any(accept_mask)
        p_out = jnp.where(any_accept, p_arr[best_idx], p0)

        # Adaptive Marquardt damping per source.
        damp_new = jnp.where(any_accept, damp_s / damp_down_j, damp_s * damp_up_j)
        damp_new = jnp.clip(damp_new, damp_min_j, damp_max_j)
        return p_out, damp_new

    def step_all(
        pos_x, pos_y, fluxes, damp_arr,
        pos_prior_mean_x, pos_prior_mean_y, pos_prior_inv_sigma,
        ix_top, iy_top, local_neighbor_idx, local_neighbor_mask,
        data_stack, invvar_stack, pix_good_stack, bg_basis_grid, bg_coefs,
        stamps_stack, psf_coefs_stack, x_origins, x_scales, y_origins, y_scales,
        grid_x0_j, grid_y0_j, max_step_pix_j,
        damp_min_j, damp_max_j, damp_up_j, damp_down_j,
    ):
        # Map only the per-source axis (axis 0); per-band stacks and
        # scalars are broadcast (in_axes=None).
        vmapped = jax.vmap(
            per_source_step,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     None, None, None, None, None, None, None, None,
                     None, None, None, None, None,
                     None, None, None, None, None, None, None),
        )
        # Per-source (axis-0) args.
        mapped = (
            jnp.arange(pos_x.shape[0]),
            pos_x, pos_y, fluxes, damp_arr,
            pos_prior_mean_x, pos_prior_mean_y, pos_prior_inv_sigma,
            ix_top, iy_top, local_neighbor_idx, local_neighbor_mask,
        )
        # Broadcast (in_axes=None) args.
        broadcast = (
            data_stack, invvar_stack, pix_good_stack, bg_basis_grid, bg_coefs,
            stamps_stack, ix_top, iy_top,
            psf_coefs_stack, x_origins, x_scales, y_origins, y_scales,
            grid_x0_j, grid_y0_j, max_step_pix_j,
            damp_min_j, damp_max_j, damp_up_j, damp_down_j,
        )
        n = pos_x.shape[0]
        if _POS_STEP_CHUNK <= 0 or n <= _POS_STEP_CHUNK:
            return vmapped(*mapped, *broadcast)
        # Chunk the per-source axis: only CHUNK sources' working set is live at
        # once -> per-grid peak memory drops ~n/CHUNK. Sources are independent,
        # so this is bit-identical to the single full vmap (same per-source ops,
        # same output order). Pad the source axis up to a multiple of CHUNK
        # (padded sources are dropped after).
        CH = _POS_STEP_CHUNK
        pad = (-n) % CH
        nchunks = (n + pad) // CH

        def _pad_reshape(a):
            if pad:
                tail = jnp.broadcast_to(a[-1:], (pad,) + a.shape[1:])
                a = jnp.concatenate([a, tail], axis=0)
            return a.reshape((nchunks, CH) + a.shape[1:])

        mapped_chunked = tuple(_pad_reshape(a) for a in mapped)

        def _per_chunk(chunk_args):
            return vmapped(*chunk_args, *broadcast)

        p_out, damp_new = jax.lax.map(_per_chunk, mapped_chunked)
        p_out = p_out.reshape((nchunks * CH,) + p_out.shape[2:])[:n]
        damp_new = damp_new.reshape((nchunks * CH,) + damp_new.shape[2:])[:n]
        return p_out, damp_new

    return jax.jit(step_all)


# ---------------------------------------------------------------------------
# Stage 2: per-source position Newton step, fluxes + bg fixed.
# ---------------------------------------------------------------------------


def stage2_position_newton_step(
    multi_inputs: GridMultiBandInputs,
    pos_x: np.ndarray, pos_y: np.ndarray,
    fluxes: np.ndarray,                # (n_sources, n_bands)
    bg_coefs: np.ndarray,              # (n_bands, M)
    max_step_pix: float = 1.0,
    damp: float | np.ndarray = 1e-3,
    damp_min: float = 1e-6,
    damp_max: float = 1e3,
    damp_up: float = 10.0,
    damp_down: float = 10.0,
    bg_poly_order: int = 2,
    precomputed_raw_stamps: "list | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take ONE Newton step on each source's (x, y), with all chi^2
    evaluation done LOCALLY on that source's stamp pixels (so memory
    scales with n_sources x stamp_pixels, NOT n_sources x grid_pixels).

    For each source:
    1. Gather a padded local data/background/weight window.
    2. Render only the neighboring source stamps that overlap that window
       and subtract them to form ``data_minus_others`` locally.
       Chi^2 is evaluated only at these stamp pixels, NOT on the full grid.
    3. Per-source 2x2 Newton step using grad + hessian of the
       per-source chi^2 (function of (x, y) only).

    Returns ``(pos_x_new, pos_y_new)``, clipped to bounds.

    This avoids the previous full-grid residual render per position step.
    """
    from .bg_polynomial import make_bg_basis_jax
    from .forced_phot_sparse import render_stamps, stamp_top_left

    n_sources = multi_inputs.n_sources
    n_bands = multi_inputs.n_bands
    band_h = multi_inputs.bands[0].band_h
    band_w = multi_inputs.bands[0].band_w
    psf_jaxes = multi_inputs.newton_psf_jaxes()
    H_psf = psf_jaxes[0].H_psf
    W_psf = psf_jaxes[0].W_psf
    grid_x0 = multi_inputs.bands[0].grid_x0
    grid_y0 = multi_inputs.bands[0].grid_y0
    pos_x_jax = jnp.asarray(pos_x, dtype=jnp.float32)
    pos_y_jax = jnp.asarray(pos_y, dtype=jnp.float32)

    data_stack = jnp.stack([
        jnp.asarray(inp_b.data, dtype=jnp.float32) for inp_b in multi_inputs.bands
    ], axis=0)
    invvar_stack = jnp.stack([
        jnp.asarray(inp_b.invvar, dtype=jnp.float32) for inp_b in multi_inputs.bands
    ], axis=0)
    pix_good_stack = jnp.stack([
        jnp.asarray(inp_b.pix_good, dtype=bool) for inp_b in multi_inputs.bands
    ], axis=0)

    M = n_terms_for_order(bg_poly_order)
    bg_basis_flat = multi_inputs.bands[0].bg_basis_flat
    if bg_basis_flat is None:
        bg_basis_flat = make_bg_basis_jax(
            band_h, band_w, grid_x0, grid_y0,
            multi_inputs.bands[0].bg_x_origin, multi_inputs.bands[0].bg_x_scale,
            multi_inputs.bands[0].bg_y_origin, multi_inputs.bands[0].bg_y_scale,
            order=bg_poly_order,
        )
    bg_basis_grid = bg_basis_flat.reshape(band_h, band_w, M)
    bg_coefs_jax = jnp.asarray(bg_coefs, dtype=jnp.float32)

    # Render each source's CURRENT stamp per band. These are used both to
    # subtract neighbors from each local data window and as the current source
    # model in the trial-position line search.
    # When the caller passes precomputed_raw_stamps (the SAME iteration's
    # linear solve rendered at the SAME positions -- positions only update
    # after the step), reuse those DeviceArrays instead of re-rendering:
    # bit-identical by construction and saves ~60% of this call's host time
    # (~1.15 s x ~190 position calls/run). Consumed slots are nulled so the
    # raw buffers free band-by-band as the scaled stack builds.
    ix_top, iy_top = stamp_top_left(
        pos_x_jax, pos_y_jax, grid_x0, grid_y0, H_psf, W_psf,
    )
    stamps_per_band: list[jnp.ndarray] = []
    for b in range(n_bands):
        if precomputed_raw_stamps is not None and precomputed_raw_stamps[b] is not None:
            raw_b = precomputed_raw_stamps[b]
            precomputed_raw_stamps[b] = None
        else:
            raw_b = render_stamps(
                psf_jaxes[b], pos_x_jax, pos_y_jax, grid_x0, grid_y0,
            )  # (n_sources, H_psf, W_psf)
        stamps_b = raw_b * jnp.asarray(fluxes[:, b], dtype=jnp.float32)[:, None, None]
        stamps_per_band.append(stamps_b)
    stamps_stack = jnp.stack(stamps_per_band, axis=0)  # (n_bands, n_sources, H_psf, W_psf)

    pos_prior_mean_x_jax = jnp.asarray(multi_inputs.pos_init_x, dtype=jnp.float32)
    pos_prior_mean_y_jax = jnp.asarray(multi_inputs.pos_init_y, dtype=jnp.float32)
    pos_prior_inv_sigma_jax = jnp.asarray(
        1.0 / np.maximum(multi_inputs.pos_err, 1e-6), dtype=jnp.float32,
    )

    # Halo for trial-position-based stamp windowing. The chi^2 below
    # gathers a (H_psf x W_psf) data window CENTERED ON THE TRIAL
    # POSITION's integer pixel (not the source's current pos's
    # integer pixel), so we eliminate the 1-px discontinuity that
    # appeared when the trial position crossed a half-integer relative
    # to the source's current pos. We pre-build a padded
    # data-minus-others window of size (H_psf + 2*halo, W_psf + 2*halo)
    # centered on the source's CURRENT integer pixel, then dynamic-slice
    # the appropriate (H_psf, W_psf) sub-window from it for each trial.
    halo_pix = max(1, int(np.ceil(max_step_pix)) + 1)  # 2 px for default max_step_pix=1.0
    H_pad = H_psf + 2 * halo_pix
    W_pad = W_psf + 2 * halo_pix

    # No-truncation neighbor list for the padded local position windows.
    # The linear solve's DEFAULT_K_MAX list is intentionally capped; this one
    # must include every current stamp that can contribute to a trial window.
    ix_top_np = np.rint(np.asarray(pos_x, dtype=np.float64) - grid_x0).astype(np.int32) - W_psf // 2
    iy_top_np = np.rint(np.asarray(pos_y, dtype=np.float64) - grid_y0).astype(np.int32) - H_psf // 2
    pad_x0 = ix_top_np - halo_pix
    pad_y0 = iy_top_np - halo_pix
    pad_x1 = pad_x0 + W_pad
    pad_y1 = pad_y0 + H_pad
    stamp_x1 = ix_top_np + W_psf
    stamp_y1 = iy_top_np + H_psf
    local_neighbors: list[np.ndarray] = []
    max_local_neighbors = 1
    for s in range(n_sources):
        overlaps = (
            (ix_top_np < pad_x1[s]) & (stamp_x1 > pad_x0[s]) &
            (iy_top_np < pad_y1[s]) & (stamp_y1 > pad_y0[s])
        )
        overlaps[s] = False
        idx = np.where(overlaps)[0].astype(np.int32)
        local_neighbors.append(idx)
        max_local_neighbors = max(max_local_neighbors, int(idx.size))
    local_neighbor_idx = np.zeros((n_sources, max_local_neighbors), dtype=np.int32)
    local_neighbor_mask = np.zeros((n_sources, max_local_neighbors), dtype=bool)
    for s, idx in enumerate(local_neighbors):
        if idx.size:
            local_neighbor_idx[s, :idx.size] = idx
            local_neighbor_mask[s, :idx.size] = True
    local_neighbor_idx_jax = jnp.asarray(local_neighbor_idx)
    local_neighbor_mask_jax = jnp.asarray(local_neighbor_mask)

    can_use_poly_jit = (
        all(isinstance(p, PsfPolynomialJax) for p in psf_jaxes)
        and len({int(p.coefs.shape[-1]) for p in psf_jaxes}) == 1
        and len({(int(p.H_psf), int(p.W_psf)) for p in psf_jaxes}) == 1
    )
    if can_use_poly_jit:
        n_terms = int(psf_jaxes[0].coefs.shape[-1])
        cache_key = (
            n_bands, band_h, band_w, H_psf, W_psf, H_pad, W_pad,
            M, n_terms, int(local_neighbor_idx.shape[1]),
        )
        step_fn = _LOCAL_POSITION_STEP_CACHE.get(cache_key)
        if step_fn is None:
            step_fn = _make_local_position_step_jit(
                n_bands=n_bands, band_h=band_h, band_w=band_w,
                H_psf=H_psf, W_psf=W_psf, H_pad=H_pad, W_pad=W_pad,
                M=M, n_terms=n_terms, k_local=int(local_neighbor_idx.shape[1]),
            )
            _LOCAL_POSITION_STEP_CACHE[cache_key] = step_fn

        psf_coefs_stack = jnp.stack(
            [jnp.asarray(p.coefs, dtype=jnp.float32) for p in psf_jaxes],
            axis=0,
        )
        x_origins = jnp.asarray([float(p.x_origin) for p in psf_jaxes], dtype=jnp.float32)
        x_scales = jnp.asarray([float(p.x_scale) for p in psf_jaxes], dtype=jnp.float32)
        y_origins = jnp.asarray([float(p.y_origin) for p in psf_jaxes], dtype=jnp.float32)
        y_scales = jnp.asarray([float(p.y_scale) for p in psf_jaxes], dtype=jnp.float32)

        if np.isscalar(damp) or (hasattr(damp, "ndim") and damp.ndim == 0):
            damp_in_fast = np.full(n_sources, float(damp), dtype=np.float32)
        else:
            damp_in_fast = np.asarray(damp, dtype=np.float32)
            if damp_in_fast.shape != (n_sources,):
                raise ValueError(
                    f"damp array shape {damp_in_fast.shape} does not match n_sources={n_sources}"
                )
        damp_in_fast_jax = jnp.asarray(damp_in_fast)

        new_p, new_damp = step_fn(
            pos_x_jax, pos_y_jax, jnp.asarray(fluxes, dtype=jnp.float32),
            damp_in_fast_jax,
            pos_prior_mean_x_jax, pos_prior_mean_y_jax, pos_prior_inv_sigma_jax,
            ix_top, iy_top, local_neighbor_idx_jax, local_neighbor_mask_jax,
            data_stack, invvar_stack, pix_good_stack, bg_basis_grid, bg_coefs_jax,
            stamps_stack, psf_coefs_stack,
            x_origins, x_scales, y_origins, y_scales,
            jnp.asarray(grid_x0, dtype=jnp.float32),
            jnp.asarray(grid_y0, dtype=jnp.float32),
            jnp.asarray(max_step_pix, dtype=jnp.float32),
            jnp.float32(damp_min), jnp.float32(damp_max),
            jnp.float32(damp_up), jnp.float32(damp_down),
        )
        pos_x_new = np.clip(np.asarray(new_p[:, 0]), multi_inputs.pos_lower_x, multi_inputs.pos_upper_x)
        pos_y_new = np.clip(np.asarray(new_p[:, 1]), multi_inputs.pos_lower_y, multi_inputs.pos_upper_y)
        pos_x_new = np.where(np.isfinite(pos_x_new), pos_x_new, np.asarray(pos_x))
        pos_y_new = np.where(np.isfinite(pos_y_new), pos_y_new, np.asarray(pos_y))
        damp_out = np.asarray(new_damp, dtype=np.float32)
        damp_out = np.where(np.isfinite(damp_out), damp_out, damp_in_fast)
        return pos_x_new.astype(np.float32), pos_y_new.astype(np.float32), damp_out

    # Per-source chi^2 evaluated ONLY on the source's stamp pixels,
    # window CENTERED ON THE TRIAL position (re-gathered each chi^2
    # eval via dynamic_slice). Stamp itself is sub-pixel-shifted via
    # Lanczos with dx = x_t - round(x_t) (always in [-0.5, 0.5]).
    damp_min_jax = jnp.float32(damp_min)
    damp_max_jax = jnp.float32(damp_max)
    damp_up_jax = jnp.float32(damp_up)
    damp_down_jax = jnp.float32(damp_down)

    def per_source_step(s_idx, x, y, fluxes_s, damp_s, mux, muy, prior_inv_sigma,
                        ix_top_s, iy_top_s, nbr_idx_row, nbr_mask_row):
        # PADDED gather window around the source's CURRENT stamp.
        # dmo_pad[b, h, w] corresponds to image pixel
        #   (iy_top_s - halo_pix + h, ix_top_s - halo_pix + w).
        ky_pad = jnp.arange(H_pad) - halo_pix
        kx_pad = jnp.arange(W_pad) - halo_pix
        ys_pad = iy_top_s + ky_pad[:, None]
        xs_pad = ix_top_s + kx_pad[None, :]
        in_bounds_pad = (
            (ys_pad >= 0) & (ys_pad < band_h) &
            (xs_pad >= 0) & (xs_pad < band_w)
        )
        safe_y_pad = jnp.clip(ys_pad, 0, band_h - 1)
        safe_x_pad = jnp.clip(xs_pad, 0, band_w - 1)
        data_pad = data_stack[:, safe_y_pad, safe_x_pad]   # (n_bands, H_pad, W_pad)
        invvar_pad = invvar_stack[:, safe_y_pad, safe_x_pad]
        pixg_pad = pix_good_stack[:, safe_y_pad, safe_x_pad]

        bg_under = bg_basis_grid[safe_y_pad, safe_x_pad, :]  # (H_pad, W_pad, M)
        bg_pad = jnp.einsum("hwm,bm->bhw", bg_under, bg_coefs_jax)

        def neighbor_pad(t_idx, mask):
            stamp_t = stamps_stack[:, t_idx]  # (n_bands, H_psf, W_psf)
            off_y = iy_top[t_idx] - iy_top_s + halo_pix
            off_x = ix_top[t_idx] - ix_top_s + halo_pix
            py = jnp.arange(H_pad)[:, None]
            px = jnp.arange(W_pad)[None, :]
            sy = py - off_y
            sx = px - off_x
            in_stamp = (sy >= 0) & (sy < H_psf) & (sx >= 0) & (sx < W_psf)
            safe_sy = jnp.clip(sy, 0, H_psf - 1)
            safe_sx = jnp.clip(sx, 0, W_psf - 1)
            vals = stamp_t[:, safe_sy, safe_sx]
            return jnp.where(mask, jnp.where(in_stamp[None, :, :], vals, 0.0), 0.0)

        other_pad = jnp.sum(
            jax.vmap(neighbor_pad)(nbr_idx_row, nbr_mask_row),
            axis=0,
        )
        dmo_pad = jnp.where(
            in_bounds_pad[None, :, :], data_pad - bg_pad - other_pad, 0.0,
        )
        w_pad = jnp.where(
            in_bounds_pad[None, :, :],
            jnp.where(pixg_pad, invvar_pad, 0.0),
            0.0,
        )

        # iy_orig / ix_orig: integer pixel of the source's CURRENT pos
        # (in band-local grid coords). Used to compute the dynamic-slice
        # offset into dmo_pad below.
        iy_orig = iy_top_s + (H_psf // 2)
        ix_orig = ix_top_s + (W_psf // 2)

        def chi2_xy(p):
            x_t, y_t = p[0], p[1]
            x_grid = x_t - grid_x0
            y_grid = y_t - grid_y0
            ix_t = jax.lax.stop_gradient(jnp.round(x_grid).astype(jnp.int32))
            iy_t = jax.lax.stop_gradient(jnp.round(y_grid).astype(jnp.int32))
            dx = x_grid - ix_t.astype(jnp.float32)
            dy = y_grid - iy_t.astype(jnp.float32)

            # Gather (H_psf, W_psf) window centered on (iy_t, ix_t)
            # from dmo_pad. The stamp window at trial pos starts at
            # image (iy_t - H_psf//2, ix_t - W_psf//2). In dmo_pad
            # coords (which start at (iy_top_s - halo_pix, ix_top_s -
            # halo_pix) = (iy_orig - H_psf//2 - halo_pix, ix_orig -
            # W_psf//2 - halo_pix)), that's (iy_t - iy_orig + halo_pix,
            # ix_t - ix_orig + halo_pix).
            start_h = (iy_t - iy_orig + halo_pix).astype(jnp.int32)
            start_w = (ix_t - ix_orig + halo_pix).astype(jnp.int32)
            zero_i32 = jnp.int32(0)
            dmo_under = jax.lax.dynamic_slice(
                dmo_pad, (zero_i32, start_h, start_w),
                (n_bands, H_psf, W_psf),
            )
            w_under = jax.lax.dynamic_slice(
                w_pad, (zero_i32, start_h, start_w),
                (n_bands, H_psf, W_psf),
            )

            chi2 = jnp.float32(0.0)
            for b in range(n_bands):
                raw = evaluate_psf_at(psf_jaxes[b], x_t, y_t)
                shifted = lanczos_shift_image(raw, dx, dy)
                stamp_at_xy = shifted * fluxes_s[b]
                diff = dmo_under[b] - stamp_at_xy
                chi2 = chi2 + jnp.sum(diff * diff * w_under[b])
            chi2 = chi2 + (prior_inv_sigma ** 2) * (
                (x_t - mux) ** 2 + (y_t - muy) ** 2
            )
            return chi2

        p0 = jnp.array([x, y], dtype=jnp.float32)
        chi2_old = chi2_xy(p0)
        g = jax.grad(chi2_xy)(p0)
        H = jax.hessian(chi2_xy)(p0)
        diag_H = jnp.diag(H)
        diag_safe = jnp.where(jnp.abs(diag_H) > 1e-30, jnp.abs(diag_H), 1e-30)
        D = jnp.float32(1.0) / jnp.sqrt(diag_safe)
        H_scaled = H * D[:, None] * D[None, :]
        g_scaled = g * D
        H_damped = H_scaled + damp_s * jnp.eye(2, dtype=jnp.float32)
        H64 = H_damped.astype(jnp.float64)
        g64 = g_scaled.astype(jnp.float64)
        delta_scaled = jnp.linalg.solve(H64, -g64).astype(jnp.float32)
        delta = D * delta_scaled
        delta = jnp.clip(delta, -max_step_pix, max_step_pix)
        delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))

        # Tractor-style line search: 11 log-spaced alphas (1/1024 .. 1).
        alphas = jnp.asarray(2.0 ** np.arange(-10, 1), dtype=jnp.float32)
        chi2_arr, p_arr = jax.vmap(lambda a: (chi2_xy(p0 + a * delta), p0 + a * delta))(alphas)
        accept_mask = chi2_arr < chi2_old
        masked_chi2 = jnp.where(accept_mask, chi2_arr, jnp.inf)
        best_idx = jnp.argmin(masked_chi2)
        any_accept = jnp.any(accept_mask)
        p_out = jnp.where(any_accept, p_arr[best_idx], p0)
        damp_new = jnp.where(any_accept, damp_s / damp_down_jax, damp_s * damp_up_jax)
        damp_new = jnp.clip(damp_new, damp_min_jax, damp_max_jax)
        return p_out, damp_new

    if np.isscalar(damp) or (hasattr(damp, "ndim") and damp.ndim == 0):
        damp_in = np.full(n_sources, float(damp), dtype=np.float32)
    else:
        damp_in = np.asarray(damp, dtype=np.float32)
        if damp_in.shape != (n_sources,):
            raise ValueError(
                f"damp array shape {damp_in.shape} does not match n_sources={n_sources}"
            )
    damp_in_jax = jnp.asarray(damp_in)

    new_p, new_damp = jax.vmap(
        per_source_step, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        jnp.arange(n_sources),
        pos_x_jax, pos_y_jax,
        jnp.asarray(fluxes, dtype=jnp.float32),
        damp_in_jax,
        pos_prior_mean_x_jax, pos_prior_mean_y_jax, pos_prior_inv_sigma_jax,
        ix_top, iy_top,
        local_neighbor_idx_jax, local_neighbor_mask_jax,
    )

    pos_x_new = np.clip(np.asarray(new_p[:, 0]), multi_inputs.pos_lower_x, multi_inputs.pos_upper_x)
    pos_y_new = np.clip(np.asarray(new_p[:, 1]), multi_inputs.pos_lower_y, multi_inputs.pos_upper_y)
    pos_x_new = np.where(np.isfinite(pos_x_new), pos_x_new, np.asarray(pos_x))
    pos_y_new = np.where(np.isfinite(pos_y_new), pos_y_new, np.asarray(pos_y))
    damp_out = np.asarray(new_damp, dtype=np.float32)
    damp_out = np.where(np.isfinite(damp_out), damp_out, damp_in)
    return pos_x_new.astype(np.float32), pos_y_new.astype(np.float32), damp_out


def stage2_source_newton_step(
    multi_inputs: GridMultiBandInputs,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
    fluxes: np.ndarray,
    bg_coefs: np.ndarray,
    max_step_pix: float = 1.0,
    max_shape_step: float = 0.15,
    direct_search_step_pix: float | None = None,
    direct_search_exp_only: bool = True,
    damp: float | np.ndarray = 1e-3,
    damp_min: float = 1e-6,
    damp_max: float = 1e3,
    damp_up: float = 10.0,
    damp_down: float = 10.0,
    bg_poly_order: int = 2,
    data_override: "list | None" = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Take one local Newton step for mixed star/exp source parameters.

    Stars update only ``x,y``. Exp sources update ``x,y,logre,ee1,ee2``.
    The fluxes and background are frozen during this step, matching the
    alternating structure used by the existing grid solver.

    Adaptive damping (Marquardt's heuristic): ``damp`` may be a scalar
    (broadcast to all sources) or a per-source array. After each step,
    each source's damp is multiplied by ``damp_up`` if its line-search
    rejected the Newton step (so the next call's H_damped is more
    diagonal-dominant -> safer step), or divided by ``damp_down`` if
    the step was accepted (so the next call moves faster). The new
    per-source damp array is returned alongside the parameters and is
    clamped to ``[damp_min, damp_max]``.
    """
    from .bg_polynomial import make_bg_basis_jax
    from .forced_phot_sparse import stamp_top_left

    n_sources = multi_inputs.n_sources
    n_bands = multi_inputs.n_bands
    band_h = multi_inputs.bands[0].band_h
    band_w = multi_inputs.bands[0].band_w
    psf_jaxes = multi_inputs.newton_psf_jaxes()
    H_psf = psf_jaxes[0].H_psf
    W_psf = psf_jaxes[0].W_psf
    grid_x0 = multi_inputs.bands[0].grid_x0
    grid_y0 = multi_inputs.bands[0].grid_y0

    source_type_np = (
        np.asarray(multi_inputs.source_type, dtype=np.int32)
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=np.int32)
    )
    source_type_jax = jnp.asarray(source_type_np, dtype=jnp.int32)
    sersic_n_np = (
        np.asarray(multi_inputs.sersic_n, dtype=np.float32)
        if multi_inputs.sersic_n is not None
        else np.ones(n_sources, dtype=np.float32)
    )
    sersic_n_jax = jnp.asarray(sersic_n_np, dtype=jnp.float32)
    render_backend = multi_inputs.render_backend
    pos_x_jax = jnp.asarray(pos_x, dtype=jnp.float32)
    pos_y_jax = jnp.asarray(pos_y, dtype=jnp.float32)
    logre_jax = jnp.asarray(logre, dtype=jnp.float32)
    ee1_jax = jnp.asarray(ee1, dtype=jnp.float32)
    ee2_jax = jnp.asarray(ee2, dtype=jnp.float32)

    # Wing peeling: fit the native model against (data - wing) when given.
    data_stack = jnp.stack([
        jnp.asarray(d, dtype=jnp.float32)
        for d in (
            data_override if data_override is not None
            else [inp_b.data for inp_b in multi_inputs.bands]
        )
    ], axis=0)
    invvar_stack = jnp.stack([
        jnp.asarray(inp_b.invvar, dtype=jnp.float32) for inp_b in multi_inputs.bands
    ], axis=0)
    pix_good_stack = jnp.stack([
        jnp.asarray(inp_b.pix_good, dtype=bool) for inp_b in multi_inputs.bands
    ], axis=0)

    M = n_terms_for_order(bg_poly_order)
    bg_basis_flat = multi_inputs.bands[0].bg_basis_flat
    if bg_basis_flat is None:
        bg_basis_flat = make_bg_basis_jax(
            band_h, band_w, grid_x0, grid_y0,
            multi_inputs.bands[0].bg_x_origin, multi_inputs.bands[0].bg_x_scale,
            multi_inputs.bands[0].bg_y_origin, multi_inputs.bands[0].bg_y_scale,
            order=bg_poly_order,
        )
    bg_basis_grid = bg_basis_flat.reshape(band_h, band_w, M)
    bg_coefs_jax = jnp.asarray(bg_coefs, dtype=jnp.float32)

    ix_top, iy_top = stamp_top_left(
        pos_x_jax, pos_y_jax, grid_x0, grid_y0, H_psf, W_psf,
    )
    stamps_per_band: list[jnp.ndarray] = []
    for b in range(n_bands):
        stamps_b = render_source_stamps(
            psf_jaxes[b], pos_x_jax, pos_y_jax, grid_x0, grid_y0,
            source_type_jax, logre_jax, ee1_jax, ee2_jax,
            sersic_n=sersic_n_jax, render_backend=render_backend,
        )
        stamps_b = stamps_b * jnp.asarray(fluxes[:, b], dtype=jnp.float32)[:, None, None]
        stamps_per_band.append(stamps_b)
    stamps_stack = jnp.stack(stamps_per_band, axis=0)

    pos_prior_mean_x_jax = jnp.asarray(multi_inputs.pos_init_x, dtype=jnp.float32)
    pos_prior_mean_y_jax = jnp.asarray(multi_inputs.pos_init_y, dtype=jnp.float32)
    pos_prior_inv_sigma_jax = jnp.asarray(
        1.0 / np.maximum(multi_inputs.pos_err, 1e-6), dtype=jnp.float32,
    )
    logre_prior = jnp.asarray(
        multi_inputs.logre_init if multi_inputs.logre_init is not None else logre,
        dtype=jnp.float32,
    )
    ee1_prior = jnp.asarray(
        multi_inputs.ee1_init if multi_inputs.ee1_init is not None else ee1,
        dtype=jnp.float32,
    )
    ee2_prior = jnp.asarray(
        multi_inputs.ee2_init if multi_inputs.ee2_init is not None else ee2,
        dtype=jnp.float32,
    )
    # (2026-07-08) The hardcoded weak shape prior (shape_prior_inv_sigma =
    # 1/10, seed-anchored) was REMOVED: input-catalog shapes are SEEDS,
    # not beliefs. logre_prior/ee1_prior/ee2_prior below stay as the seed
    # values used to hold non-EXP shape params fixed in the mixed Newton
    # step (NOT as a Gaussian pull).

    logre_lower = jnp.asarray(
        multi_inputs.logre_lower
        if multi_inputs.logre_lower is not None
        else np.full(n_sources, np.log(0.1), dtype=np.float32),
        dtype=jnp.float32,
    )
    logre_upper = jnp.asarray(
        multi_inputs.logre_upper
        if multi_inputs.logre_upper is not None
        else np.full(n_sources, np.log(1000.0), dtype=np.float32),
        dtype=jnp.float32,
    )
    ee1_lower = jnp.asarray(
        multi_inputs.ee1_lower
        if multi_inputs.ee1_lower is not None
        else np.full(n_sources, -100.0, dtype=np.float32),
        dtype=jnp.float32,
    )
    ee1_upper = jnp.asarray(
        multi_inputs.ee1_upper
        if multi_inputs.ee1_upper is not None
        else np.full(n_sources, 100.0, dtype=np.float32),
        dtype=jnp.float32,
    )
    ee2_lower = jnp.asarray(
        multi_inputs.ee2_lower
        if multi_inputs.ee2_lower is not None
        else np.full(n_sources, -100.0, dtype=np.float32),
        dtype=jnp.float32,
    )
    ee2_upper = jnp.asarray(
        multi_inputs.ee2_upper
        if multi_inputs.ee2_upper is not None
        else np.full(n_sources, 100.0, dtype=np.float32),
        dtype=jnp.float32,
    )

    direct_search_step_pix = float(max_step_pix if direct_search_step_pix is None else direct_search_step_pix)
    halo_pix = max(1, int(np.ceil(max(max_step_pix, direct_search_step_pix))) + 1)
    H_pad = H_psf + 2 * halo_pix
    W_pad = W_psf + 2 * halo_pix

    ix_top_np = np.rint(np.asarray(pos_x, dtype=np.float64) - grid_x0).astype(np.int32) - W_psf // 2
    iy_top_np = np.rint(np.asarray(pos_y, dtype=np.float64) - grid_y0).astype(np.int32) - H_psf // 2
    pad_x0 = ix_top_np - halo_pix
    pad_y0 = iy_top_np - halo_pix
    pad_x1 = pad_x0 + W_pad
    pad_y1 = pad_y0 + H_pad
    stamp_x1 = ix_top_np + W_psf
    stamp_y1 = iy_top_np + H_psf
    local_neighbors: list[np.ndarray] = []
    max_local_neighbors = 1
    for s in range(n_sources):
        overlaps = (
            (ix_top_np < pad_x1[s]) & (stamp_x1 > pad_x0[s]) &
            (iy_top_np < pad_y1[s]) & (stamp_y1 > pad_y0[s])
        )
        overlaps[s] = False
        idx = np.where(overlaps)[0].astype(np.int32)
        local_neighbors.append(idx)
        max_local_neighbors = max(max_local_neighbors, int(idx.size))
    local_neighbor_idx = np.zeros((n_sources, max_local_neighbors), dtype=np.int32)
    local_neighbor_mask = np.zeros((n_sources, max_local_neighbors), dtype=bool)
    for s, idx in enumerate(local_neighbors):
        if idx.size:
            local_neighbor_idx[s, :idx.size] = idx
            local_neighbor_mask[s, :idx.size] = True
    local_neighbor_idx_jax = jnp.asarray(local_neighbor_idx)
    local_neighbor_mask_jax = jnp.asarray(local_neighbor_mask)

    ky_pad = jnp.arange(H_pad) - halo_pix
    kx_pad = jnp.arange(W_pad) - halo_pix
    py_pad = jnp.arange(H_pad)[:, None]
    px_pad = jnp.arange(W_pad)[None, :]
    eye5 = jnp.eye(5, dtype=jnp.float32)

    damp_min_jax = jnp.float32(damp_min)
    damp_max_jax = jnp.float32(damp_max)
    damp_up_jax = jnp.float32(damp_up)
    damp_down_jax = jnp.float32(damp_down)

    def per_source_step(
        s_idx, x, y, lr, e1, e2, stype, sn_t, fluxes_s, damp_s,
        mux, muy, prior_inv_sigma, lr0, e10, e20,
        plx, pux, ply, puy, lrl, lru, e1l, e1u, e2l, e2u,
        ix_top_s, iy_top_s, nbr_idx_row, nbr_mask_row,
    ):
        ys_pad = iy_top_s + ky_pad[:, None]
        xs_pad = ix_top_s + kx_pad[None, :]
        in_bounds_pad = (
            (ys_pad >= 0) & (ys_pad < band_h) &
            (xs_pad >= 0) & (xs_pad < band_w)
        )
        safe_y_pad = jnp.clip(ys_pad, 0, band_h - 1)
        safe_x_pad = jnp.clip(xs_pad, 0, band_w - 1)
        data_pad = data_stack[:, safe_y_pad, safe_x_pad]
        invvar_pad = invvar_stack[:, safe_y_pad, safe_x_pad]
        pixg_pad = pix_good_stack[:, safe_y_pad, safe_x_pad]
        bg_under = bg_basis_grid[safe_y_pad, safe_x_pad, :]
        bg_pad = jnp.einsum("hwm,bm->bhw", bg_under, bg_coefs_jax)

        def neighbor_pad(t_idx, mask):
            stamp_t = stamps_stack[:, t_idx]
            off_y = iy_top[t_idx] - iy_top_s + halo_pix
            off_x = ix_top[t_idx] - ix_top_s + halo_pix
            sy = py_pad - off_y
            sx = px_pad - off_x
            in_stamp = (sy >= 0) & (sy < H_psf) & (sx >= 0) & (sx < W_psf)
            safe_sy = jnp.clip(sy, 0, H_psf - 1)
            safe_sx = jnp.clip(sx, 0, W_psf - 1)
            vals = stamp_t[:, safe_sy, safe_sx]
            return jnp.where(mask, jnp.where(in_stamp[None, :, :], vals, 0.0), 0.0)

        other_pad = jnp.sum(jax.vmap(neighbor_pad)(nbr_idx_row, nbr_mask_row), axis=0)
        dmo_pad = jnp.where(in_bounds_pad[None, :, :], data_pad - bg_pad - other_pad, 0.0)
        w_pad = jnp.where(
            in_bounds_pad[None, :, :],
            jnp.where(pixg_pad, invvar_pad, 0.0),
            0.0,
        )
        iy_orig = iy_top_s + (H_psf // 2)
        ix_orig = ix_top_s + (W_psf // 2)
        is_exp = stype == SOURCE_EXP

        def clip_params(p):
            return jnp.array([
                jnp.clip(p[0], plx, pux),
                jnp.clip(p[1], ply, puy),
                jnp.where(is_exp, jnp.clip(p[2], lrl, lru), lr0),
                jnp.where(is_exp, jnp.clip(p[3], e1l, e1u), e10),
                jnp.where(is_exp, jnp.clip(p[4], e2l, e2u), e20),
            ], dtype=jnp.float32)

        def chi2_params(p_in):
            p = clip_params(p_in)
            x_t, y_t, lr_t, e1_t, e2_t = p
            x_grid = x_t - grid_x0
            y_grid = y_t - grid_y0
            ix_t = jax.lax.stop_gradient(jnp.round(x_grid).astype(jnp.int32))
            iy_t = jax.lax.stop_gradient(jnp.round(y_grid).astype(jnp.int32))
            start_h = (iy_t - iy_orig + halo_pix).astype(jnp.int32)
            start_w = (ix_t - ix_orig + halo_pix).astype(jnp.int32)
            zero_i32 = jnp.int32(0)
            dmo_under = jax.lax.dynamic_slice(
                dmo_pad, (zero_i32, start_h, start_w),
                (n_bands, H_psf, W_psf),
            )
            w_under = jax.lax.dynamic_slice(
                w_pad, (zero_i32, start_h, start_w),
                (n_bands, H_psf, W_psf),
            )
            chi2 = jnp.float32(0.0)
            for b in range(n_bands):
                stamp = render_source_stamp_at(
                    psf_jaxes[b], x_t, y_t, grid_x0, grid_y0,
                    stype, lr_t, e1_t, e2_t,
                    sersic_n=sn_t, render_backend=render_backend,
                )
                diff = dmo_under[b] - stamp * fluxes_s[b]
                chi2 = chi2 + jnp.sum(diff * diff * w_under[b])
            chi2 = chi2 + (prior_inv_sigma ** 2) * (
                (x_t - mux) ** 2 + (y_t - muy) ** 2
            )
            # NOTE (2026-07-08): the hardcoded 1/10 seed-anchoring shape
            # prior was REMOVED here. A seed is a starting point, not a
            # belief -- absent a catalog-supplied width there is NO shape
            # prior (matches the CPU path + the island solver's opt-in
            # *_PRIOR_SIG mechanism). The position prior above stays (it
            # is gated by the opt-in POS_ERR sigma). lr0/e10/e20 remain in
            # use above to HOLD non-EXP shape params at their seed.
            return chi2

        p0 = jnp.array([x, y, lr, e1, e2], dtype=jnp.float32)
        chi2_old, g = jax.value_and_grad(chi2_params)(p0)
        H = jax.hessian(chi2_params)(p0)

        # Column scaling (Marquardt's parameter normalization). Whitens
        # the 5-D parameter space so position (in pixels), logre (in
        # log-pixels), and ee1/ee2 (dimensionless) all carry comparable
        # weight in the linear solve. Standard form:
        #   D = diag(1/sqrt(H_ii))                    (per-param scale)
        #   delta = D * solve(D H D + lambda*I, D g)
        # This lifts the conditioning of H by orders of magnitude when
        # parameters have very different curvature scales, which is
        # routinely the case for galaxy EXP shapes.
        diag_H = jnp.diag(H)
        diag_safe = jnp.where(jnp.abs(diag_H) > 1e-30, jnp.abs(diag_H), 1e-30)
        D = jnp.float32(1.0) / jnp.sqrt(diag_safe)
        H_scaled = H * D[:, None] * D[None, :]
        g_scaled = g * D
        H_damped = H_scaled + damp_s * eye5
        H64 = H_damped.astype(jnp.float64)
        g64 = g_scaled.astype(jnp.float64)
        delta_scaled = jnp.linalg.solve(H64, -g64).astype(jnp.float32)
        delta = D * delta_scaled
        step_clip = jnp.array([
            max_step_pix, max_step_pix,
            max_shape_step, max_shape_step, max_shape_step,
        ], dtype=jnp.float32)
        delta = jnp.clip(delta, -step_clip, step_clip)
        delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
        delta = jnp.where(is_exp, delta, delta.at[2:].set(0.0))

        # Mirror tractor.optimize.Optimizer.tryUpdates' wide log-spaced
        # alpha grid (1/1024 .. 1, factors of 2). The line search picks
        # the BEST chi^2-improving alpha. With ~3 orders of magnitude
        # of granularity it can effectively zero out a wrong-direction
        # Newton step without triggering the direct-search fallback.
        #
        # We DON'T include Tractor's alpha=sqrt(2)/2.0 probes because
        # our delta is already clipped to +/- max_step_pix before the
        # line search; allowing alpha > 1 would silently let the step
        # exceed the configured cap and reintroduces the overshoot
        # this fix is meant to prevent.
        alphas = jnp.asarray(2.0 ** np.arange(-10, 1), dtype=jnp.float32)

        def alpha_chi2(alpha):
            p_trial = clip_params(p0 + alpha * delta)
            return chi2_params(p_trial), p_trial

        chi2_arr, p_arr = jax.vmap(alpha_chi2)(alphas)
        # Reject any probe that worsens chi^2 (would correspond to
        # Tractor's `pAfter < pBest - 1` cutoff but tighter -- we want
        # strict improvement).
        accept_mask = chi2_arr < chi2_old
        masked_chi2 = jnp.where(accept_mask, chi2_arr, jnp.inf)
        best_idx = jnp.argmin(masked_chi2)
        any_accept = jnp.any(accept_mask)
        chi2_curr = jnp.where(any_accept, chi2_arr[best_idx], chi2_old)
        p_curr = jnp.where(any_accept, p_arr[best_idx], clip_params(p0))

        # If the local Newton direction is poor, try a small direct-search
        # stencil.  This is important for EXP rows: the 5-D Hessian can be
        # badly conditioned at catalog seeds, while a one-pixel trial move is
        # often enough to reveal the same descent direction Tractor finds.
        s = jnp.asarray(direct_search_step_pix, dtype=jnp.float32)
        hs = jnp.float32(0.5) * max_step_pix
        zero = jnp.float32(0.0)
        xy_offsets = jnp.array([
            [zero, zero, zero, zero, zero],
            [s, zero, zero, zero, zero],
            [-s, zero, zero, zero, zero],
            [zero, s, zero, zero, zero],
            [zero, -s, zero, zero, zero],
            [s, s, zero, zero, zero],
            [s, -s, zero, zero, zero],
            [-s, s, zero, zero, zero],
            [-s, -s, zero, zero, zero],
            [hs, zero, zero, zero, zero],
            [-hs, zero, zero, zero, zero],
            [zero, hs, zero, zero, zero],
            [zero, -hs, zero, zero, zero],
        ], dtype=jnp.float32)
        if direct_search_exp_only:
            xy_offsets = jnp.where(is_exp, xy_offsets, jnp.zeros_like(xy_offsets))
        candidates = jax.vmap(lambda off: clip_params(p0 + off))(xy_offsets)
        candidate_chi2 = jax.vmap(chi2_params)(candidates)
        best_i = jnp.argmin(candidate_chi2)
        best_p = candidates[best_i]
        best_chi2 = candidate_chi2[best_i]
        # Only fall back to the direct-search probes when the Newton step's
        # line search failed to improve chi^2 (i.e., chi2_curr is still the
        # initial chi2_old).  Always running the probes overrode Newton's
        # accurate small steps for faint sources with shallow chi^2
        # surfaces, allowing the position to drift to noise pockets.
        newton_succeeded = chi2_curr < chi2_old
        use_fallback = (~newton_succeeded) & (best_chi2 < chi2_curr)
        p_out = jnp.where(use_fallback, best_p, p_curr)

        # Adaptive damping (Marquardt). Definition of "step accepted":
        # either the Newton+line-search produced a chi^2 reduction, OR
        # the direct-search fallback did. Otherwise the step was a no-op
        # and we should stiffen damp for the next iteration.
        step_accepted = newton_succeeded | use_fallback
        damp_new = jnp.where(
            step_accepted, damp_s / damp_down_jax, damp_s * damp_up_jax,
        )
        damp_new = jnp.clip(damp_new, damp_min_jax, damp_max_jax)
        # chi2 IMPROVEMENT achieved by this step (0 when every candidate was
        # rejected). Landscape-drift-immune (chi2_old and the best candidate
        # are evaluated on the SAME flux/neighbor state), so the outer S3
        # loop can apply a per-source dlnp-style ftol on it directly.
        chi2_best = jnp.where(use_fallback, best_chi2, chi2_curr)
        chi2_gain = jnp.maximum(chi2_old - chi2_best, 0.0)
        return p_out, damp_new, chi2_gain

    if np.isscalar(damp) or (hasattr(damp, "ndim") and damp.ndim == 0):
        damp_in = np.full(n_sources, float(damp), dtype=np.float32)
    else:
        damp_in = np.asarray(damp, dtype=np.float32)
        if damp_in.shape != (n_sources,):
            raise ValueError(
                f"damp array shape {damp_in.shape} does not match n_sources={n_sources}"
            )
    damp_in_jax = jnp.asarray(damp_in)

    vmapped_step = jax.vmap(
        per_source_step,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0),
    )
    # All 30 args are per-source (axis 0); the per-band image stacks are
    # closed over by per_source_step and shared across sources.
    mapped_args = (
        jnp.arange(n_sources),
        pos_x_jax, pos_y_jax, logre_jax, ee1_jax, ee2_jax, source_type_jax,
        sersic_n_jax,
        jnp.asarray(fluxes, dtype=jnp.float32),
        damp_in_jax,
        pos_prior_mean_x_jax, pos_prior_mean_y_jax, pos_prior_inv_sigma_jax,
        logre_prior, ee1_prior, ee2_prior,
        jnp.asarray(multi_inputs.pos_lower_x, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_upper_x, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_lower_y, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_upper_y, dtype=jnp.float32),
        logre_lower, logre_upper, ee1_lower, ee1_upper, ee2_lower, ee2_upper,
        ix_top, iy_top, local_neighbor_idx_jax, local_neighbor_mask_jax,
    )

    # ---- Memory-guarded chunking of the mixed-Newton vmap ----
    # A single full vmap materializes every source's padded neighbour
    # windows at once. The DOMINANT buffer is the per-neighbour gather
    # (n_sources, n_bands, k_local, H_pad, W_pad): fine at UDS scale
    # (~600 src x 4 bands x k_local~20 ~ 1 GB) but astronomically large
    # at crowded-field scale -- T08285 + extended is 1800 src x 20 bands
    # x k_local~190 (bulge core) x 59^2, i.e. ~96 GB for a full vmap and
    # still 25.5 GiB for a fixed 512-source chunk. Mirror the star
    # path's TRACT7DT_POS_STEP_CHUNK lax.map chunking, but with a
    # K_LOCAL-AWARE trigger + chunk size: chunk only when the estimated
    # full-vmap gather exceeds _SRC_STEP_CHUNK_TRIGGER_BYTES (existing
    # small-field runs keep the exact single-vmap program, bit-compat),
    # and size the chunk so the gather stays under
    # _SRC_STEP_CHUNK_BUDGET_BYTES. Crowded fields trade the documented
    # fp trajectory-class change (vmap vs lax.map lowering, within noise
    # -- June validation) for a bounded peak. TRACT7DT_POS_STEP_CHUNK=0
    # forces the legacy single vmap everywhere (may OOM on crowded
    # extended fields).
    k_local = int(local_neighbor_idx_jax.shape[1]) if local_neighbor_idx_jax.ndim > 1 else 1
    per_source_bytes = (
        n_bands * max(k_local, 1) * H_pad * W_pad * 4 * _SRC_STEP_AD_COPIES
    )
    full_bytes = n_sources * per_source_bytes
    use_chunk = _SRC_STEP_CHUNK_CAP > 0 and full_bytes > _SRC_STEP_CHUNK_TRIGGER_BYTES
    if use_chunk:
        CH = int(max(16, min(
            _SRC_STEP_CHUNK_CAP,
            _SRC_STEP_CHUNK_BUDGET_BYTES // per_source_bytes,
        )))
        use_chunk = n_sources > CH
    if not use_chunk:
        new_p, new_damp, new_chi2 = vmapped_step(*mapped_args)
    else:
        pad = (-n_sources) % CH
        nchunks = (n_sources + pad) // CH

        def _pad_reshape(a):
            if pad:
                tail = jnp.broadcast_to(a[-1:], (pad,) + a.shape[1:])
                a = jnp.concatenate([a, tail], axis=0)
            return a.reshape((nchunks, CH) + a.shape[1:])

        mapped_chunked = tuple(_pad_reshape(a) for a in mapped_args)
        new_p, new_damp, new_chi2 = jax.lax.map(
            lambda chunk_args: vmapped_step(*chunk_args), mapped_chunked,
        )
        new_p = new_p.reshape((nchunks * CH,) + new_p.shape[2:])[:n_sources]
        new_damp = new_damp.reshape(nchunks * CH)[:n_sources]
        new_chi2 = new_chi2.reshape(nchunks * CH)[:n_sources]

    out = np.asarray(new_p)
    damp_out = np.asarray(new_damp, dtype=np.float32)
    pos_x_new = np.where(np.isfinite(out[:, 0]), out[:, 0], np.asarray(pos_x))
    pos_y_new = np.where(np.isfinite(out[:, 1]), out[:, 1], np.asarray(pos_y))
    logre_new = np.where(np.isfinite(out[:, 2]), out[:, 2], np.asarray(logre))
    ee1_new = np.where(np.isfinite(out[:, 3]), out[:, 3], np.asarray(ee1))
    ee2_new = np.where(np.isfinite(out[:, 4]), out[:, 4], np.asarray(ee2))
    is_exp_np = source_type_np == SOURCE_EXP
    logre_new = np.where(is_exp_np, logre_new, np.asarray(logre))
    ee1_new = np.where(is_exp_np, ee1_new, np.asarray(ee1))
    ee2_new = np.where(is_exp_np, ee2_new, np.asarray(ee2))
    damp_out = np.where(np.isfinite(damp_out), damp_out, damp_in)
    chi2_gain_out = np.asarray(new_chi2, dtype=np.float64)
    return (
        pos_x_new.astype(np.float32),
        pos_y_new.astype(np.float32),
        logre_new.astype(np.float32),
        ee1_new.astype(np.float32),
        ee2_new.astype(np.float32),
        damp_out,
        chi2_gain_out,
    )


# ---------------------------------------------------------------------------
# Wing peeling for large galaxies.
#
# With the native 55px stamp, wings of galaxies with nre(n)*Re beyond the
# render window are absorbed by the background model (measured: -10..-14%
# flux at Re 16-40 px; free-n fits of cuspy n>=4 land 0.7-0.9 low). The
# whole-grid stamp-tier promotion fix OOM'd (per-source Newton windows and
# sparse neighbour gathers scale ~tier^2). Wing peeling instead keeps ALL
# solver machinery at the native stamp and maintains per-band WING images
# for the few "big" sources -- the flux-scaled difference between the
# same renderer evaluated at a zero-padded tier-T PSF and at the native
# PSF, using identical placement conventions (stamp_top_left + Lanczos
# shift) -- and subtracts them from the data before every linear solve and
# position/shape step:  (data - wing) - native_model == data - full_model,
# with no double counting. The wing is recomputed at each S3 outer
# iteration from the CURRENT fluxes/shapes/n (fixed-point iteration).
# ---------------------------------------------------------------------------


def _pad_psf_jax(psf_jax, tier: int):
    """Zero-pad a PsfPolynomialJax stamp to (tier, tier).

    Same zero-pad convention as runner._maybe_pad_psf (kept separate to
    avoid a circular import): the PSF stays compact and unit-sum; only
    the render box grows. No-op when the tier does not exceed the
    native size or for PSF objects without a ``coefs`` array.
    """
    import dataclasses
    H = int(getattr(psf_jax, "H_psf", 0))
    W = int(getattr(psf_jax, "W_psf", 0))
    coefs = getattr(psf_jax, "coefs", None)
    if coefs is None or (tier <= H and tier <= W):
        return psf_jax
    coefs_np = np.asarray(coefs)
    pad_y0 = (tier - H) // 2
    pad_x0 = (tier - W) // 2
    new_coefs = np.zeros((tier, tier) + coefs_np.shape[2:], dtype=coefs_np.dtype)
    new_coefs[pad_y0:pad_y0 + H, pad_x0:pad_x0 + W] = coefs_np
    return dataclasses.replace(
        psf_jax, coefs=jnp.asarray(new_coefs), H_psf=tier, W_psf=tier)


def _wing_nre(sersic_n: np.ndarray) -> np.ndarray:
    """Render-box requirement in effective radii (Tractor patch convention:
    4 for disky n<1.5, 6 for 1.5<=n<3, 8 for cuspy n>=3)."""
    n = np.asarray(sersic_n, dtype=np.float64)
    return np.where(n < 1.5, 4.0, np.where(n < 3.0, 6.0, 8.0))


def _select_wing_sources(
    source_type: np.ndarray,
    logre: np.ndarray,
    sersic_n: np.ndarray,
    w_psf_native: int,
    max_sources: int,
    log_fn=None,
) -> np.ndarray:
    """Indices of "big" sources whose wings need peeling.

    big = extended (non-STAR) AND nre(n) * Re_current > 0.5 * W_psf_native.
    Capped at ``max_sources`` per grid (keep the largest by nre*Re)."""
    is_ext = np.asarray(source_type, dtype=np.int32) != SOURCE_STAR
    req_half = _wing_nre(sersic_n) * np.exp(np.asarray(logre, dtype=np.float64))
    big = is_ext & np.isfinite(req_half) & (req_half > 0.5 * float(w_psf_native))
    idx = np.where(big)[0]
    if idx.size > int(max_sources):
        if log_fn is not None:
            log_fn(
                f"wing peel: {idx.size} big sources exceed the cap "
                f"{int(max_sources)}; peeling only the largest by nre*Re"
            )
        order = np.argsort(-req_half[idx])
        idx = np.sort(idx[order[: int(max_sources)]])
    return idx.astype(np.int64)


def _scatter_stamps_to_band(
    stamps: jnp.ndarray,      # (n, S, S)
    ix_top: jnp.ndarray,      # (n,) int32
    iy_top: jnp.ndarray,      # (n,) int32
    band_h: int,
    band_w: int,
) -> jnp.ndarray:             # (band_h, band_w)
    """Accumulate variable-size stamps into a band image (bounds-clipped);
    same math as the joint step's accumulate_stamps_one_band."""
    n, S_h, S_w = int(stamps.shape[0]), int(stamps.shape[1]), int(stamps.shape[2])
    py = jnp.arange(S_h, dtype=jnp.int32)
    px = jnp.arange(S_w, dtype=jnp.int32)
    iy_idx = jnp.broadcast_to(iy_top[:, None, None] + py[None, :, None], (n, S_h, S_w))
    ix_idx = jnp.broadcast_to(ix_top[:, None, None] + px[None, None, :], (n, S_h, S_w))
    valid = (
        (iy_idx >= 0) & (iy_idx < band_h) & (ix_idx >= 0) & (ix_idx < band_w)
    )
    iy_safe = jnp.clip(iy_idx, 0, band_h - 1)
    ix_safe = jnp.clip(ix_idx, 0, band_w - 1)
    vals = jnp.where(valid, stamps, 0.0).reshape(-1)
    return (
        jnp.zeros((band_h, band_w), dtype=stamps.dtype)
        .at[iy_safe.reshape(-1), ix_safe.reshape(-1)]
        .add(vals)
    )


def _compute_wing_images(
    multi_inputs: "GridMultiBandInputs",
    psf_native_jaxes: list,
    psf_wing_jaxes: list,
    big_idx: np.ndarray,
    pos_x: np.ndarray, pos_y: np.ndarray,
    logre: np.ndarray, ee1: np.ndarray, ee2: np.ndarray,
    sersic_n: np.ndarray,
    fluxes: np.ndarray,            # (n_sources, n_bands)
) -> list:
    """Per-band wing images for the "big" sources.

    wing_b = sum_big flux_s,b * [scatter(stamp_T(s)) - scatter(stamp_native(s))]

    Both renders use the pipeline's own render_source_stamps with the SAME
    placement (stamp_top_left uses round(pos)-size//2, both sizes odd), so
    subtracting the wing from the data and fitting the native model is
    exactly equivalent to fitting the full tier-T model for big sources.
    """
    from .forced_phot_sparse import stamp_top_left

    band_h = multi_inputs.bands[0].band_h
    band_w = multi_inputs.bands[0].band_w
    grid_x0 = multi_inputs.bands[0].grid_x0
    grid_y0 = multi_inputs.bands[0].grid_y0
    render_backend = multi_inputs.render_backend

    px_b = jnp.asarray(np.asarray(pos_x)[big_idx], dtype=jnp.float32)
    py_b = jnp.asarray(np.asarray(pos_y)[big_idx], dtype=jnp.float32)
    st_b = jnp.asarray(
        np.asarray(multi_inputs.source_type, dtype=np.int32)[big_idx])
    lr_b = jnp.asarray(np.asarray(logre)[big_idx], dtype=jnp.float32)
    e1_b = jnp.asarray(np.asarray(ee1)[big_idx], dtype=jnp.float32)
    e2_b = jnp.asarray(np.asarray(ee2)[big_idx], dtype=jnp.float32)
    sn_b = jnp.asarray(np.asarray(sersic_n)[big_idx], dtype=jnp.float32)
    flux_b = np.asarray(fluxes, dtype=np.float32)[big_idx, :]
    # Non-finite fluxes (unfit) contribute no wing.
    flux_b = np.where(np.isfinite(flux_b), flux_b, 0.0)

    wings = []
    for b in range(multi_inputs.n_bands):
        wing_pair = []
        for psf_jax in (psf_wing_jaxes[b], psf_native_jaxes[b]):
            S = int(psf_jax.H_psf)
            stamps = render_source_stamps(
                psf_jax, px_b, py_b, grid_x0, grid_y0,
                st_b, lr_b, e1_b, e2_b,
                sersic_n=sn_b, render_backend=render_backend,
            )
            stamps = stamps * jnp.asarray(flux_b[:, b])[:, None, None]
            ix_top, iy_top = stamp_top_left(
                px_b, py_b, grid_x0, grid_y0, S, S,
            )
            wing_pair.append(_scatter_stamps_to_band(
                stamps, ix_top, iy_top, band_h, band_w))
        wings.append(wing_pair[0] - wing_pair[1])   # tier - native
    return wings


# ---------------------------------------------------------------------------
# Joint LSQR position step (new optional path)
# ---------------------------------------------------------------------------


_JOINT_STEP_JIT_CACHE: dict[tuple, "jax.stages.Wrapped"] = {}
_VARPRO_STEP_JIT_CACHE: dict[tuple, "jax.stages.Wrapped"] = {}
_VARPRO_NERR_JIT_CACHE: dict[tuple, "jax.stages.Wrapped"] = {}


def _joint_varpro_step(
    multi_inputs: GridMultiBandInputs,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
    fluxes_init: np.ndarray,         # (n_sources, n_bands) -- for preconditioner
    neighbor_idx: np.ndarray,
    neighbor_mask: np.ndarray,
    *,
    pos_step_max_pix: float,
    max_shape_step: float = 0.15,
    damp: float,
    cg_max_iters: int,
    cg_tol: float,
    use_preconditioner: bool,
    bg_poly_order: int,
    reg: float = 1e-9,
    sersic_n: np.ndarray | None = None,   # (n_sources,) current n state
    free_n: np.ndarray | None = None,     # (n_sources,) bool -- fit n for these
    data_override: "list | None" = None,  # per-band (data - wing) images
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """Joint LM step over (X, F, B) via variable projection.

    Returns ``(pos_x, pos_y, logre, ee1, ee2, sersic_n, fluxes, bg_coefs)``.

    Unlike :func:`_joint_position_shape_step` (which only solves X
    jointly while F/B stay frozen between alternating S1 calls), this
    step computes the OPTIMAL F*, B* at every residual evaluation
    INSIDE the residual function.  JAX autodiff through the closed-form
    ``jnp.linalg.solve`` propagates the chain rule, so the LM step
    direction is identical to a full joint LSQR over (F, X, B).

    Stage-2 free-n: the parameter vector is 6 wide per source --
    ``(x, y, logre, ee1, ee2, u_n)`` with ``u_n`` the logit-bijected
    Sersic index. Rows with ``free_n`` True (TYPE=SERSIC passing the
    S/N gate) fit ``n``; every other row's ``u_n`` is masked out of the
    step and renders from the frozen ``sersic_n`` value, so grids
    without free-n rows behave exactly like the 5-parameter fit.
    """
    from .bg_polynomial import make_bg_basis_jax
    from .joint_lsqr import (
        cheap_diagonal_preconditioner, make_joint_position_step_fn, make_varpro_residual_fn,
    )
    from .forced_phot_sparse import _make_solve_band_jit

    n_sources = multi_inputs.n_sources
    n_bands = multi_inputs.n_bands
    band_h = multi_inputs.bands[0].band_h
    band_w = multi_inputs.bands[0].band_w
    psf_jaxes = multi_inputs.newton_psf_jaxes()
    H_psf = psf_jaxes[0].H_psf
    W_psf = psf_jaxes[0].W_psf
    grid_x0 = multi_inputs.bands[0].grid_x0
    grid_y0 = multi_inputs.bands[0].grid_y0

    M = n_terms_for_order(bg_poly_order)
    bg_basis_flat = multi_inputs.bands[0].bg_basis_flat
    if bg_basis_flat is None:
        bg_basis_flat = make_bg_basis_jax(
            band_h, band_w, grid_x0, grid_y0,
            multi_inputs.bands[0].bg_x_origin, multi_inputs.bands[0].bg_x_scale,
            multi_inputs.bands[0].bg_y_origin, multi_inputs.bands[0].bg_y_scale,
            order=bg_poly_order,
        )
    bg_basis_grid = bg_basis_flat.reshape(band_h, band_w, M)

    # Wing peeling: the varpro residual (and the final flux solve below)
    # fit the native model against (data - wing) when given.
    data_stack = jnp.stack([
        jnp.asarray(d, dtype=jnp.float32)
        for d in (
            data_override if data_override is not None
            else [b.data for b in multi_inputs.bands]
        )
    ], axis=0)
    invvar_stack = jnp.stack([
        jnp.asarray(b.invvar, dtype=jnp.float32) for b in multi_inputs.bands
    ], axis=0)
    pix_good_stack = jnp.stack([
        jnp.asarray(b.pix_good, dtype=bool) for b in multi_inputs.bands
    ], axis=0)
    source_type_jax = jnp.asarray(multi_inputs.source_type, dtype=jnp.int32) \
        if multi_inputs.source_type is not None \
        else jnp.zeros(n_sources, dtype=jnp.int32)
    # Current per-source Sersic-n STATE (fitted rows update across steps;
    # falls back to the catalog seeds carried on multi_inputs).
    if sersic_n is None:
        sersic_n = (
            np.asarray(multi_inputs.sersic_n, dtype=np.float32)
            if multi_inputs.sersic_n is not None
            else np.ones(n_sources, dtype=np.float32)
        )
    sersic_n = np.asarray(sersic_n, dtype=np.float32)
    if free_n is None:
        free_n = np.zeros(n_sources, dtype=bool)
    free_n = np.asarray(free_n, dtype=bool)
    sersic_n_jax = jnp.asarray(sersic_n, dtype=jnp.float32)
    free_n_jax = jnp.asarray(free_n, dtype=bool)
    # u_n seeds via the logit bijection (frozen rows carry an inert value).
    u_seed = sersic_n_to_u(sersic_n).astype(np.float32)
    render_backend = multi_inputs.render_backend

    nbr_idx_jax = jnp.asarray(neighbor_idx, dtype=jnp.int32)
    nbr_mask_jax = jnp.asarray(neighbor_mask, dtype=bool)

    residual_fn = make_varpro_residual_fn(
        sersic_n=sersic_n_jax,
        render_backend=render_backend,
        data_stack=data_stack,
        invvar_stack=invvar_stack,
        pix_good_stack=pix_good_stack,
        bg_basis_grid=bg_basis_grid,
        psf_jaxes=psf_jaxes,
        grid_x0=grid_x0, grid_y0=grid_y0,
        source_type=source_type_jax,
        neighbor_idx=nbr_idx_jax,
        neighbor_mask=nbr_mask_jax,
        H_psf=H_psf, W_psf=W_psf,
        band_h=band_h, band_w=band_w,
        bg_poly_order=bg_poly_order,
        reg=float(reg),
        free_n_mask=free_n_jax,
    )
    # ALL per-grid data (image stacks, bg basis, neighbor lists, source
    # types, sersic_n, grid origin) flow through this traced pytree into
    # the cached jitted step. Only true statics (shapes, render_backend,
    # reg, the per-band PSF models -- identical across grids in a run)
    # live in residual_fn.core's closure, so one compiled step program
    # is valid for every same-shape grid. (Previously the closed-over
    # data stacks were baked into the XLA program as constants and a
    # second same-shape grid was silently fit against the first grid's
    # data.)
    varpro_arrays = residual_fn.arrays
    residual_core = residual_fn.core

    cache_key = (
        n_sources, n_bands, band_h, band_w, H_psf, W_psf, M,
        int(neighbor_idx.shape[1]),
        int(cg_max_iters), bool(use_preconditioner), str(render_backend),
    )
    step_fn = _VARPRO_STEP_JIT_CACHE.get(cache_key)
    if step_fn is None:
        step_fn = make_joint_position_step_fn(
            residual_core, n_sources,
            pos_step_max_pix=pos_step_max_pix,
            max_shape_step=max_shape_step,
            cg_max_iters=int(cg_max_iters),
            cg_tol=float(cg_tol),
            use_preconditioner=bool(use_preconditioner),
            n_params_per_source=6,
        )
        step_fn = jax.jit(step_fn)
        _VARPRO_STEP_JIT_CACHE[cache_key] = step_fn

    p_init = jnp.stack([
        jnp.asarray(pos_x, dtype=jnp.float32),
        jnp.asarray(pos_y, dtype=jnp.float32),
        jnp.asarray(logre, dtype=jnp.float32),
        jnp.asarray(ee1, dtype=jnp.float32),
        jnp.asarray(ee2, dtype=jnp.float32),
        jnp.asarray(u_seed, dtype=jnp.float32),
    ], axis=1).reshape(-1)

    if use_preconditioner:
        # Cheap per-source-local approximation; ~O(n_sources) cost
        # vs diagonal_JTJ's O(n_params) full-grid scan.
        is_ext_np = (np.asarray(multi_inputs.source_type, dtype=np.int32) != SOURCE_STAR)
        logre_safe_np = np.where(is_ext_np, logre, 0.0).astype(np.float32)
        ee1_safe_np = np.where(is_ext_np, ee1, 1.0).astype(np.float32)
        ee2_safe_np = np.where(is_ext_np, ee2, 0.0).astype(np.float32)
        diag = cheap_diagonal_preconditioner(
            sersic_n=sersic_n_jax,
            render_backend=render_backend,
            fluxes=jnp.asarray(fluxes_init, dtype=jnp.float32),
            invvar_stack=invvar_stack,
            pix_good_stack=pix_good_stack,
            psf_jaxes=psf_jaxes,
            pos_x=jnp.asarray(pos_x, dtype=jnp.float32),
            pos_y=jnp.asarray(pos_y, dtype=jnp.float32),
            grid_x0=grid_x0, grid_y0=grid_y0,
            source_type=source_type_jax,
            logre=jnp.asarray(logre_safe_np),
            ee1=jnp.asarray(ee1_safe_np),
            ee2=jnp.asarray(ee2_safe_np),
            H_psf=H_psf, W_psf=W_psf,
            band_h=band_h, band_w=band_w,
            fit_n=True,
            free_n_mask=free_n_jax,
            u_n=jnp.asarray(u_seed, dtype=jnp.float32),
        )
    else:
        diag = jnp.zeros_like(p_init)

    # NOTE: varpro_arrays is the traced per-grid data pytree; it MUST be
    # passed every call so the shape-cached step_fn sees THIS grid's data.
    p_new, _chi2_old, _chi2_new, _alpha_used, _delta_norm = step_fn(
        p_init,
        source_type_jax,
        jnp.asarray(multi_inputs.pos_lower_x, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_upper_x, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_lower_y, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_upper_y, dtype=jnp.float32),
        jnp.asarray(multi_inputs.logre_lower, dtype=jnp.float32)
            if multi_inputs.logre_lower is not None
            else jnp.full(n_sources, np.log(0.1), dtype=jnp.float32),
        jnp.asarray(multi_inputs.logre_upper, dtype=jnp.float32)
            if multi_inputs.logre_upper is not None
            else jnp.full(n_sources, np.log(1000.0), dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee1_lower, dtype=jnp.float32)
            if multi_inputs.ee1_lower is not None
            else jnp.full(n_sources, -100.0, dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee1_upper, dtype=jnp.float32)
            if multi_inputs.ee1_upper is not None
            else jnp.full(n_sources, 100.0, dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee2_lower, dtype=jnp.float32)
            if multi_inputs.ee2_lower is not None
            else jnp.full(n_sources, -100.0, dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee2_upper, dtype=jnp.float32)
            if multi_inputs.ee2_upper is not None
            else jnp.full(n_sources, 100.0, dtype=jnp.float32),
        jnp.float32(damp),
        diag,
        varpro_arrays,
        # u_n hard bounds: the logit bijection already soft-bounds n to
        # (0.29, 6.3); |u| <= 12 keeps the sigmoid numerically sane.
        jnp.full(n_sources, -12.0, dtype=jnp.float32),
        jnp.full(n_sources, 12.0, dtype=jnp.float32),
    )

    out = np.asarray(p_new).reshape(n_sources, 6)
    pos_x_new = np.where(np.isfinite(out[:, 0]), out[:, 0], np.asarray(pos_x))
    pos_y_new = np.where(np.isfinite(out[:, 1]), out[:, 1], np.asarray(pos_y))
    logre_new = np.where(np.isfinite(out[:, 2]), out[:, 2], np.asarray(logre))
    ee1_new = np.where(np.isfinite(out[:, 3]), out[:, 3], np.asarray(ee1))
    ee2_new = np.where(np.isfinite(out[:, 4]), out[:, 4], np.asarray(ee2))
    u_new = np.where(np.isfinite(out[:, 5]), out[:, 5], u_seed)
    # Shape updates apply to ALL extended rows (EXP/DEV/SERSIC) -- the
    # 6-param step's keep_mask frees (logre, ee1, ee2) for every
    # non-STAR row (Stage-2 behavior; the old 5-param step was EXP-only).
    is_ext_gate_np = (
        np.asarray(multi_inputs.source_type, dtype=np.int32) != SOURCE_STAR
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=bool)
    )
    logre_new = np.where(is_ext_gate_np, logre_new, np.asarray(logre))
    ee1_new = np.where(is_ext_gate_np, ee1_new, np.asarray(ee1))
    ee2_new = np.where(is_ext_gate_np, ee2_new, np.asarray(ee2))
    # Sersic-n update only for the free rows; everyone else keeps state.
    sersic_n_new = np.where(
        free_n, sersic_u_to_n(u_new).astype(np.float32), sersic_n,
    ).astype(np.float32)

    # Now compute the OPTIMAL fluxes / bg coefs at the new positions.
    # We can re-use the JIT'd per-band solver from forced_phot_sparse.
    fluxes = np.zeros((n_sources, n_bands), dtype=np.float32)
    bg_coefs = np.zeros((n_bands, M), dtype=np.float32)
    flux_var = np.zeros((n_sources, n_bands), dtype=np.float32)
    bg_var = np.zeros((n_bands, M), dtype=np.float32)
    _solver_unjit, solver_jit = _make_solve_band_jit(
        H_psf=H_psf, W_psf=W_psf,
        band_h=band_h, band_w=band_w,
        n_sources=n_sources, k_max=int(neighbor_idx.shape[1]),
        M=M,
    )
    pos_x_jax = jnp.asarray(pos_x_new, dtype=jnp.float32)
    pos_y_jax = jnp.asarray(pos_y_new, dtype=jnp.float32)
    # Render-safe shapes: EXTENDED sources (EXP/DEV/SERSIC) keep their true
    # shape values; stars get inert placeholders. Note is_exp_np above only
    # gates which sources' shapes were UPDATED by the step (stage-1: EXP);
    # DEV/SERSIC render at their frozen seed shapes.
    is_ext_np = (
        np.asarray(multi_inputs.source_type, dtype=np.int32) != SOURCE_STAR
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=bool)
    )
    logre_jax = jnp.asarray(np.where(is_ext_np, logre_new, 0.0), dtype=jnp.float32)
    ee1_jax = jnp.asarray(np.where(is_ext_np, ee1_new, 1.0), dtype=jnp.float32)
    ee2_jax = jnp.asarray(np.where(is_ext_np, ee2_new, 0.0), dtype=jnp.float32)
    # Render with the UPDATED Sersic-n state (fitted rows moved above).
    sersic_n_new_jax = jnp.asarray(sersic_n_new, dtype=jnp.float32)
    from .forced_phot_sparse import stamp_top_left
    ix_top, iy_top = stamp_top_left(
        pos_x_jax, pos_y_jax, grid_x0, grid_y0, H_psf, W_psf,
    )
    for b in range(n_bands):
        stamps_b = render_source_stamps(
            psf_jaxes[b], pos_x_jax, pos_y_jax, grid_x0, grid_y0,
            source_type_jax, logre_jax, ee1_jax, ee2_jax,
            sersic_n=sersic_n_new_jax, render_backend=render_backend,
        )
        F_b, B_b, Fvar_b, Bvar_b = solver_jit(
            data_stack[b], invvar_stack[b], pix_good_stack[b],
            stamps_b, ix_top, iy_top,
            nbr_idx_jax, nbr_mask_jax,
            bg_basis_flat, jnp.float32(reg),
        )
        fluxes[:, b] = np.asarray(F_b)
        bg_coefs[b, :] = np.asarray(B_b)
        flux_var[:, b] = np.asarray(Fvar_b)
        bg_var[b, :] = np.asarray(Bvar_b)

    return (
        pos_x_new.astype(np.float32),
        pos_y_new.astype(np.float32),
        logre_new.astype(np.float32),
        ee1_new.astype(np.float32),
        ee2_new.astype(np.float32),
        sersic_n_new,
        fluxes,
        bg_coefs,
    )


def _varpro_sersic_n_err(
    multi_inputs: GridMultiBandInputs,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
    sersic_n: np.ndarray,
    free_n: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_mask: np.ndarray,
    *,
    bg_poly_order: int,
    reg: float = 1e-9,
    data_override: "list | None" = None,
) -> np.ndarray:
    """Per-source 1-sigma uncertainty of the FITTED Sersic index.

    At the converged parameters, for each free-n source s::

        J_s     = d r / d u_n[s]   (jvp of the varpro residual along the
                                    unit vector of s's u_n entry)
        sigma_u = 1 / ||J_s||_2
        sigma_n = sigma_u * |dn/du|,
        dn/du   = (n - n_min) * (n_max - n) / (n_max - n_min)   (logit bijection)

    The varpro residual is already sqrt(invvar)-weighted and FLUX/BG-
    MARGINALIZED (the closed-form linear solve lives inside it, so the
    Golub-Pereyra chain rule of F*/B* w.r.t. u_n is included in the jvp).
    It is a DIAGONAL estimate: the n <-> position/shape covariance (the
    Schur off-diagonal terms) is ignored, so the error is conservative-ish
    for blended or shape-degenerate cases. Frozen rows return NaN.
    """
    from .bg_polynomial import make_bg_basis_jax
    from .joint_lsqr import make_varpro_residual_fn
    from .source_model import SERSIC_N_MAX, SERSIC_N_MIN

    n_sources = multi_inputs.n_sources
    n_bands = multi_inputs.n_bands
    err = np.full(n_sources, np.nan, dtype=np.float32)
    free_n = np.asarray(free_n, dtype=bool)
    free_idx = np.where(free_n)[0]
    if free_idx.size == 0:
        return err

    band_h = multi_inputs.bands[0].band_h
    band_w = multi_inputs.bands[0].band_w
    psf_jaxes = multi_inputs.newton_psf_jaxes()
    H_psf = psf_jaxes[0].H_psf
    W_psf = psf_jaxes[0].W_psf
    grid_x0 = multi_inputs.bands[0].grid_x0
    grid_y0 = multi_inputs.bands[0].grid_y0
    M = n_terms_for_order(bg_poly_order)
    bg_basis_flat = multi_inputs.bands[0].bg_basis_flat
    if bg_basis_flat is None:
        bg_basis_flat = make_bg_basis_jax(
            band_h, band_w, grid_x0, grid_y0,
            multi_inputs.bands[0].bg_x_origin, multi_inputs.bands[0].bg_x_scale,
            multi_inputs.bands[0].bg_y_origin, multi_inputs.bands[0].bg_y_scale,
            order=bg_poly_order,
        )
    bg_basis_grid = bg_basis_flat.reshape(band_h, band_w, M)

    data_stack = jnp.stack([
        jnp.asarray(d, dtype=jnp.float32)
        for d in (
            data_override if data_override is not None
            else [b.data for b in multi_inputs.bands]
        )
    ], axis=0)
    invvar_stack = jnp.stack([
        jnp.asarray(b.invvar, dtype=jnp.float32) for b in multi_inputs.bands
    ], axis=0)
    pix_good_stack = jnp.stack([
        jnp.asarray(b.pix_good, dtype=bool) for b in multi_inputs.bands
    ], axis=0)
    source_type_jax = jnp.asarray(multi_inputs.source_type, dtype=jnp.int32) \
        if multi_inputs.source_type is not None \
        else jnp.zeros(n_sources, dtype=jnp.int32)
    sersic_n = np.asarray(sersic_n, dtype=np.float32)
    render_backend = multi_inputs.render_backend

    residual_fn = make_varpro_residual_fn(
        sersic_n=jnp.asarray(sersic_n),
        render_backend=render_backend,
        data_stack=data_stack,
        invvar_stack=invvar_stack,
        pix_good_stack=pix_good_stack,
        bg_basis_grid=bg_basis_grid,
        psf_jaxes=psf_jaxes,
        grid_x0=grid_x0, grid_y0=grid_y0,
        source_type=source_type_jax,
        neighbor_idx=jnp.asarray(neighbor_idx, dtype=jnp.int32),
        neighbor_mask=jnp.asarray(neighbor_mask, dtype=bool),
        H_psf=H_psf, W_psf=W_psf,
        band_h=band_h, band_w=band_w,
        bg_poly_order=bg_poly_order,
        reg=float(reg),
        free_n_mask=jnp.asarray(free_n),
    )
    residual_core = residual_fn.core
    varpro_arrays = residual_fn.arrays

    # Shape-keyed jitted direction-norm; same closure contract as the step
    # cache (residual_core's closure holds only statics; ALL per-grid data
    # flow through the traced ``arrays`` pytree, and the parameter index k
    # is a traced scalar), so one compiled program serves every same-shape
    # grid and every free source.
    cache_key = (
        n_sources, n_bands, band_h, band_w, H_psf, W_psf, M,
        int(neighbor_idx.shape[1]), str(render_backend),
    )
    dir_norm = _VARPRO_NERR_JIT_CACHE.get(cache_key)
    if dir_norm is None:
        def _dir_norm(p_flat, arrays, k):
            tangent = jnp.zeros_like(p_flat).at[k].set(1.0)
            _, Jt = jax.jvp(
                lambda q: residual_core(q, arrays), (p_flat,), (tangent,))
            return jnp.sqrt(jnp.sum(Jt * Jt))
        dir_norm = jax.jit(_dir_norm)
        _VARPRO_NERR_JIT_CACHE[cache_key] = dir_norm

    u_final = sersic_n_to_u(sersic_n).astype(np.float32)
    p_final = jnp.stack([
        jnp.asarray(pos_x, dtype=jnp.float32),
        jnp.asarray(pos_y, dtype=jnp.float32),
        jnp.asarray(logre, dtype=jnp.float32),
        jnp.asarray(ee1, dtype=jnp.float32),
        jnp.asarray(ee2, dtype=jnp.float32),
        jnp.asarray(u_final),
    ], axis=1).reshape(-1)

    # Sequential loop keeps memory at one residual buffer (vmapping ~40
    # directions would hold n_free full-grid residuals at once).
    dndu = (
        (sersic_n - SERSIC_N_MIN) * (SERSIC_N_MAX - sersic_n)
        / (SERSIC_N_MAX - SERSIC_N_MIN)
    ).astype(np.float64)
    for s in free_idx:
        norm = float(dir_norm(p_final, varpro_arrays, jnp.int32(s * 6 + 5)))
        if np.isfinite(norm) and norm > 0:
            err[s] = np.float32((1.0 / norm) * abs(dndu[s]))
    return err


def _joint_position_shape_step(
    multi_inputs: GridMultiBandInputs,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
    fluxes: np.ndarray,                # (n_sources, n_bands)
    bg_coefs: np.ndarray,              # (n_bands, M)
    *,
    pos_step_max_pix: float,
    max_shape_step: float = 0.15,
    damp: float,
    cg_max_iters: int,
    cg_tol: float,
    use_preconditioner: bool,
    bg_poly_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Joint LSQR / preconditioned CG step over the WHOLE grid.

    Replaces ``stage2_source_newton_step``'s per-source Gauss-Seidel
    Newton with a single CG-on-normal-equations solve that captures the
    full inter-source off-diagonal Hessian coupling.

    Returns ``(pos_x, pos_y, logre, ee1, ee2)`` with the same dtype as
    the per-source path (no ``damp_arr`` because joint LSQR uses a
    single scalar damping handled at the LM level).
    """
    from .bg_polynomial import make_bg_basis_jax
    from .joint_lsqr import (
        diagonal_JTJ, make_joint_position_step_fn, make_residual_fn,
    )

    n_sources = multi_inputs.n_sources
    n_bands = multi_inputs.n_bands
    band_h = multi_inputs.bands[0].band_h
    band_w = multi_inputs.bands[0].band_w
    psf_jaxes = multi_inputs.newton_psf_jaxes()
    H_psf = psf_jaxes[0].H_psf
    W_psf = psf_jaxes[0].W_psf
    grid_x0 = multi_inputs.bands[0].grid_x0
    grid_y0 = multi_inputs.bands[0].grid_y0

    M = n_terms_for_order(bg_poly_order)
    bg_basis_flat = multi_inputs.bands[0].bg_basis_flat
    if bg_basis_flat is None:
        bg_basis_flat = make_bg_basis_jax(
            band_h, band_w, grid_x0, grid_y0,
            multi_inputs.bands[0].bg_x_origin, multi_inputs.bands[0].bg_x_scale,
            multi_inputs.bands[0].bg_y_origin, multi_inputs.bands[0].bg_y_scale,
            order=bg_poly_order,
        )
    bg_basis_grid = bg_basis_flat.reshape(band_h, band_w, M)

    data_stack = jnp.stack([
        jnp.asarray(b.data, dtype=jnp.float32) for b in multi_inputs.bands
    ], axis=0)
    invvar_stack = jnp.stack([
        jnp.asarray(b.invvar, dtype=jnp.float32) for b in multi_inputs.bands
    ], axis=0)
    pix_good_stack = jnp.stack([
        jnp.asarray(b.pix_good, dtype=bool) for b in multi_inputs.bands
    ], axis=0)

    source_type_jax = jnp.asarray(multi_inputs.source_type, dtype=jnp.int32) \
        if multi_inputs.source_type is not None \
        else jnp.zeros(n_sources, dtype=jnp.int32)
    sersic_n_jax = jnp.asarray(multi_inputs.sersic_n, dtype=jnp.float32) \
        if multi_inputs.sersic_n is not None \
        else jnp.ones(n_sources, dtype=jnp.float32)
    render_backend = multi_inputs.render_backend

    residual_fn = make_residual_fn(
        sersic_n=sersic_n_jax,
        render_backend=render_backend,
        data_stack=data_stack,
        invvar_stack=invvar_stack,
        pix_good_stack=pix_good_stack,
        bg_basis_grid=bg_basis_grid,
        bg_coefs=jnp.asarray(bg_coefs, dtype=jnp.float32),
        fluxes=jnp.asarray(fluxes, dtype=jnp.float32),
        psf_jaxes=psf_jaxes,
        grid_x0=grid_x0,
        grid_y0=grid_y0,
        source_type=source_type_jax,
        H_psf=H_psf,
        W_psf=W_psf,
        band_h=band_h,
        band_w=band_w,
    )
    # ALL per-grid data (image stacks, bg basis, FROZEN fluxes/bg coefs,
    # source types, sersic_n, grid origin) flow through this traced
    # pytree into the cached jitted step. Only true statics (shapes,
    # render_backend, the per-band PSF models -- identical across grids
    # in a run) live in residual_fn.core's closure, so one compiled step
    # program is valid for every same-shape grid AND for later
    # iterations of the same grid with refit fluxes/bg. (Previously the
    # closed-over data + fluxes were baked into the XLA program as
    # constants: a second same-shape grid -- or the same grid's next S3
    # iteration -- was silently fit against the first call's data.)
    joint_arrays = residual_fn.arrays
    residual_core = residual_fn.core
    # Per-call jit of the BOUND closure: used only for the diagonal-JTJ
    # preconditioner below, which is rebuilt every call (never cached),
    # so the baked-in data are this call's own.
    residual_fn_jit = jax.jit(residual_fn)

    # JIT-cache the joint step closure on the grid shape signature.
    cache_key = (
        n_sources, n_bands, band_h, band_w, H_psf, W_psf, M,
        int(cg_max_iters), bool(use_preconditioner), str(render_backend),
    )
    step_fn = _JOINT_STEP_JIT_CACHE.get(cache_key)
    if step_fn is None:
        step_fn = make_joint_position_step_fn(
            residual_core, n_sources,
            pos_step_max_pix=pos_step_max_pix,
            max_shape_step=max_shape_step,
            cg_max_iters=int(cg_max_iters),
            cg_tol=float(cg_tol),
            use_preconditioner=bool(use_preconditioner),
        )
        step_fn = jax.jit(step_fn)
        _JOINT_STEP_JIT_CACHE[cache_key] = step_fn

    p_init = jnp.stack([
        jnp.asarray(pos_x, dtype=jnp.float32),
        jnp.asarray(pos_y, dtype=jnp.float32),
        jnp.asarray(logre, dtype=jnp.float32),
        jnp.asarray(ee1, dtype=jnp.float32),
        jnp.asarray(ee2, dtype=jnp.float32),
    ], axis=1).reshape(-1)

    if use_preconditioner:
        diag = diagonal_JTJ(residual_fn_jit, p_init)
    else:
        diag = jnp.zeros_like(p_init)

    # NOTE: joint_arrays is the traced per-grid data pytree; it MUST be
    # passed every call so the shape-cached step_fn sees THIS grid's data.
    p_new, _chi2_old, _chi2_new, _alpha_used, _delta_norm = step_fn(
        p_init,
        source_type_jax,
        jnp.asarray(multi_inputs.pos_lower_x, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_upper_x, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_lower_y, dtype=jnp.float32),
        jnp.asarray(multi_inputs.pos_upper_y, dtype=jnp.float32),
        jnp.asarray(multi_inputs.logre_lower, dtype=jnp.float32)
            if multi_inputs.logre_lower is not None
            else jnp.full(n_sources, np.log(0.1), dtype=jnp.float32),
        jnp.asarray(multi_inputs.logre_upper, dtype=jnp.float32)
            if multi_inputs.logre_upper is not None
            else jnp.full(n_sources, np.log(1000.0), dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee1_lower, dtype=jnp.float32)
            if multi_inputs.ee1_lower is not None
            else jnp.full(n_sources, -100.0, dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee1_upper, dtype=jnp.float32)
            if multi_inputs.ee1_upper is not None
            else jnp.full(n_sources, 100.0, dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee2_lower, dtype=jnp.float32)
            if multi_inputs.ee2_lower is not None
            else jnp.full(n_sources, -100.0, dtype=jnp.float32),
        jnp.asarray(multi_inputs.ee2_upper, dtype=jnp.float32)
            if multi_inputs.ee2_upper is not None
            else jnp.full(n_sources, 100.0, dtype=jnp.float32),
        jnp.float32(damp),
        diag,
        joint_arrays,
    )

    out = np.asarray(p_new).reshape(n_sources, 5)
    pos_x_new = np.where(np.isfinite(out[:, 0]), out[:, 0], np.asarray(pos_x))
    pos_y_new = np.where(np.isfinite(out[:, 1]), out[:, 1], np.asarray(pos_y))
    logre_new = np.where(np.isfinite(out[:, 2]), out[:, 2], np.asarray(logre))
    ee1_new = np.where(np.isfinite(out[:, 3]), out[:, 3], np.asarray(ee1))
    ee2_new = np.where(np.isfinite(out[:, 4]), out[:, 4], np.asarray(ee2))
    is_exp_np = (
        np.asarray(multi_inputs.source_type, dtype=np.int32) == SOURCE_EXP
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=bool)
    )
    logre_new = np.where(is_exp_np, logre_new, np.asarray(logre))
    ee1_new = np.where(is_exp_np, ee1_new, np.asarray(ee1))
    ee2_new = np.where(is_exp_np, ee2_new, np.asarray(ee2))
    return (
        pos_x_new.astype(np.float32),
        pos_y_new.astype(np.float32),
        logre_new.astype(np.float32),
        ee1_new.astype(np.float32),
        ee2_new.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Top-level: 3-stage LM solver for ONE grid.
# ---------------------------------------------------------------------------


def solve_grid_3stage(
    multi_inputs: GridMultiBandInputs,
    s2_max_iters: int = 1,
    s3_max_outer_iters: int = 5,
    pos_step_max_pix: float = 1.0,
    pos_dlnp_tol_pix: float = 0.01,
    k_max_neighbors: int = DEFAULT_K_MAX,
    reg: float = 1e-9,
    bg_poly_order: int = 2,
    single_stage: bool = False,
    blank_response_correction: bool = False,
    n_opt_iters: int | None = None,
    position_solver: str = "per_source",
    joint_cg_max_iters: int = 20,
    joint_cg_tol: float = 1e-6,
    joint_use_preconditioner: bool = True,
    pos_stop_unconverged_frac: float = 0.0,
    chi2_ftol: float = 0.1,
    fit_sersic_n: bool = False,
    n_fit_min_snr: float = 30.0,
    wing_peeling: bool = False,
    wing_peel_tier: int = 221,
    wing_peel_max_sources: int = 64,
    log_fn: Optional[Callable[[str], None]] = None,
) -> GridMultiBandResult:
    """Run the LM fit on one image grid.

    ``wing_peeling`` (large-galaxy wings): keep the solver at the native
    stamp but subtract per-band WING images (flux x [tier-T render minus
    native render]) of the few "big" sources (nre(n)*Re > half the native
    stamp, capped at ``wing_peel_max_sources``) from the data before every
    linear solve and position/shape step. The wing is refreshed at each S3
    outer iteration from the current fluxes/shapes/n. Star-only grids never
    peel (big selection requires extended sources).

    ``fit_sersic_n`` / ``n_fit_min_snr`` (Stage-2, varpro only): free the
    Sersic index of TYPE=SERSIC sources whose S1 flux S/N proxy
    (median over bands of |F|/sqrt(F_var)) reaches ``n_fit_min_snr``;
    lower-S/N SERSIC rows keep their catalog seed and are reported with
    ``sersic_n_frozen=True`` (free-n is unconstrained for faint sources).
    Ignored by the per_source and joint_lsqr position solvers.

    ``pos_stop_unconverged_frac``: S3 quantile stop. 0 (default) = exact
    legacy behavior (every source must step below tolerance). >0 declares
    S3 converged when at most ``floor(frac * n_sources)`` sources are still
    above tolerance -- on T08285 the last 1-4 'walker' sources per grid
    (catalog positions off by 1-2 px, clip-limited descent) drag 55 of 145
    S3 iterations while >=99.8% of sources converge by iteration 2. The
    stragglers are reported per source in ``position_stranded`` (and as a
    catalog column) instead of poisoning grid-level convergence flags.

    Two modes (selected by ``single_stage``):

    **single_stage=False** (3-stage, mirrors per-patch CPU mode):

    - **S1**: positions FROZEN at ``multi_inputs.pos_init_x/y``;
      per-band linear sparse forced-phot solve gives initial fluxes
      + bg coefs.
    - **S2**: positions THAWED, fluxes/bg from S1 frozen. ONE Newton
      step (default) gives the first position correction.
    - **S3**: ALL thawed. Up to ``s3_max_outer_iters`` outer iterations
      of (S1-style flux+bg sparse solve + S2-style position step).

    **single_stage=True** (joint fit, "fit once and for all"):

    - Skip S1 + S2 entirely.
    - Run directly to the alternating loop with positions starting at
      the input catalog seeds. Up to ``n_opt_iters`` outer iterations
      of (linear flux+bg refit + per-source Newton position step) until
      max position update < ``pos_dlnp_tol_pix``.
    - Recommended when input catalog positions are already accurate
      (e.g. Gaia-augmented), since the staged thaw pattern is mainly
      to handle stale-position-frozen-flux pathology that doesn't apply.

    The final result holds positions/fluxes/bg coefs + their
    diagonal variances.
    """
    # Honor the new general-purpose `n_opt_iters` knob: in single_stage
    # mode it's the joint-loop iter cap; in 3-stage mode it overrides
    # s3_max_outer_iters.
    if single_stage:
        s3_max_outer_iters = int(n_opt_iters) if n_opt_iters is not None else s3_max_outer_iters
    else:
        if n_opt_iters is not None:
            s3_max_outer_iters = int(n_opt_iters)
    n_sources = multi_inputs.n_sources
    n_bands = multi_inputs.n_bands
    H_psf = multi_inputs.bands[0].psf_jax.H_psf
    W_psf = multi_inputs.bands[0].psf_jax.W_psf

    pos_x = multi_inputs.pos_init_x.copy().astype(np.float32)
    pos_y = multi_inputs.pos_init_y.copy().astype(np.float32)
    logre = (
        multi_inputs.logre_init.copy().astype(np.float32)
        if multi_inputs.logre_init is not None
        else np.zeros(n_sources, dtype=np.float32)
    )
    ee1 = (
        multi_inputs.ee1_init.copy().astype(np.float32)
        if multi_inputs.ee1_init is not None
        else np.zeros(n_sources, dtype=np.float32)
    )
    ee2 = (
        multi_inputs.ee2_init.copy().astype(np.float32)
        if multi_inputs.ee2_init is not None
        else np.zeros(n_sources, dtype=np.float32)
    )
    source_type = (
        np.asarray(multi_inputs.source_type, dtype=np.int32)
        if multi_inputs.source_type is not None
        else np.zeros(n_sources, dtype=np.int32)
    )
    # Sersic-n STATE: fitted for free-n SERSIC rows (varpro), otherwise
    # carried at the catalog seed all the way through.
    sersic_n = (
        np.asarray(multi_inputs.sersic_n, dtype=np.float32).copy()
        if multi_inputs.sersic_n is not None
        else np.ones(n_sources, dtype=np.float32)
    )
    # "has_exp" = has ANY extended source (EXP/DEV/SERSIC): routes the grid
    # onto the mixed-renderer position step instead of the star-only path.
    has_exp = bool(np.any(source_type != SOURCE_STAR))

    # Per-source adaptive damping state (Marquardt's lambda). Starts at
    # the conventional 1e-3 and adapts up/down by 10x based on whether
    # the per-source line search accepted the Newton step in S2/S3.
    # Carried across S2 and S3 outer iters so a source that's been
    # struggling stays in "small-step" mode for a few iterations.
    damp_arr = np.full(n_sources, 1e-3, dtype=np.float32)

    # Build neighbor list ONCE (positions don't change between stages
    # by enough to materially shift the neighbor structure; we rebuild
    # at the top of S3 if max position shift is large).
    neighbor_idx, neighbor_mask = build_neighbor_list(
        pos_x, pos_y, H_psf, W_psf, k_max=k_max_neighbors,
    )

    if single_stage:
        # Joint loop only. Skip S1 (positions-frozen linear solve) and
        # S2 (one position Newton step). Initialize fluxes / bg with a
        # SINGLE linear forced-phot solve at the input positions, then
        # the joint loop alternates flux-refit + position step.
        t_stage = time.perf_counter()
        s1_results, s1_ok = stage1_solve(
            multi_inputs, pos_x, pos_y, logre, ee1, ee2,
            neighbor_idx, neighbor_mask, reg=reg,
            bg_poly_order=bg_poly_order,
        )
        if log_fn is not None:
            log_fn(f"init linear solve done in {time.perf_counter() - t_stage:.2f}s")
        s2_ok = True  # not run; report as "ok" to avoid skewing summaries
    else:
        # ---- S1 ----
        t_stage = time.perf_counter()
        s1_results, s1_ok, s1_stamps = stage1_solve(
            multi_inputs, pos_x, pos_y, logre, ee1, ee2,
            neighbor_idx, neighbor_mask, reg=reg,
            bg_poly_order=bg_poly_order,
            return_stamps=True,
        )
        if log_fn is not None:
            log_fn(f"S1 linear solve done in {time.perf_counter() - t_stage:.2f}s")
    fluxes = np.stack([r.fluxes for r in s1_results], axis=1)         # (n_sources, n_bands)
    bg_coefs = np.stack([r.bg_coefs for r in s1_results], axis=0)     # (n_bands, M)
    flux_var = np.stack([r.flux_var for r in s1_results], axis=1)
    bg_var = np.stack([r.bg_var for r in s1_results], axis=0)

    # ---- Stage-2 free-n decision (varpro only), from the S1 solve ----
    # free_n = TYPE==SERSIC rows whose per-band-median flux S/N proxy
    # clears n_fit_min_snr. Below the gate the Sersic index is
    # unconstrained by the data (AstroPhot cross-fits: faint free-n rows
    # pin at the bounds), so it stays FROZEN at the catalog seed.
    free_n = np.zeros(n_sources, dtype=bool)
    if (
        fit_sersic_n
        and position_solver == "joint_varpro"
        and bool(np.any(source_type == SOURCE_SERSIC))
    ):
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_bands = np.abs(fluxes) / np.sqrt(np.maximum(flux_var, 1e-30))
        snr_proxy = np.nanmedian(snr_bands, axis=1)
        free_n = (
            (source_type == SOURCE_SERSIC)
            & np.isfinite(snr_proxy)
            & (snr_proxy >= float(n_fit_min_snr))
        )
        if log_fn is not None:
            n_ser = int(np.sum(source_type == SOURCE_SERSIC))
            log_fn(
                f"free-n gate: {int(np.sum(free_n))}/{n_ser} SERSIC rows "
                f"pass S/N >= {float(n_fit_min_snr):g} (rest frozen at seed)"
            )

    # ---- Wing peeling state (large-galaxy wings) ----
    # Star-only grids never peel: wing_active requires has_exp, and the
    # big-source selection additionally requires extended rows, so the
    # star-only bit gate is untouched (data_eff stays None).
    wing_active = bool(wing_peeling) and has_exp
    _wing_native_jaxes = multi_inputs.newton_psf_jaxes() if wing_active else None
    _wing_psf_jaxes: "list | None" = None
    _wing_logged = False

    def _wing_data_eff(fluxes_now):
        """Per-band (data - wing) from the CURRENT state; None = no peeling.

        The wing render is cheap (n_big <= wing_peel_max_sources stamps of
        wing_peel_tier^2 px per band), so it is refreshed from the current
        fluxes/shapes/n at every S3 outer iteration.
        """
        nonlocal _wing_psf_jaxes, _wing_logged
        if not wing_active:
            return None
        big_idx = _select_wing_sources(
            source_type, logre, sersic_n,
            int(_wing_native_jaxes[0].W_psf), int(wing_peel_max_sources),
            log_fn=log_fn,
        )
        if big_idx.size == 0:
            return None
        if _wing_psf_jaxes is None:
            _wing_psf_jaxes = [
                _pad_psf_jax(p, int(wing_peel_tier)) for p in _wing_native_jaxes
            ]
        if log_fn is not None and not _wing_logged:
            log_fn(
                f"wing peel: {big_idx.size} big source(s) -> tier "
                f"{int(wing_peel_tier)} wings subtracted from the data"
            )
            _wing_logged = True
        wings = _compute_wing_images(
            multi_inputs, _wing_native_jaxes, _wing_psf_jaxes, big_idx,
            pos_x, pos_y, logre, ee1, ee2, sersic_n, fluxes_now,
        )
        return [
            jnp.asarray(b.data, dtype=jnp.float32) - w
            for b, w in zip(multi_inputs.bands, wings)
        ]

    # First wing from the (unpeeled) S1 fluxes -- feeds S2; refreshed at
    # each S3 outer iteration below (fixed-point iteration on the flux).
    data_eff = _wing_data_eff(fluxes)

    if not single_stage:
        # ---- S2 (single position step from S1's flux/bg) ----
        s2_ok = False
        if log_fn is not None:
            log_fn(f"S1 done  n_sources={n_sources}  n_bands={n_bands}  bg_poly_order={bg_poly_order}")
        for it in range(s2_max_iters):
            t_stage = time.perf_counter()
            new_n = sersic_n   # only the varpro branch below updates n
            if position_solver == "joint_varpro":
                new_x, new_y, new_logre, new_ee1, new_ee2, new_n, fluxes, bg_coefs = _joint_varpro_step(
                    multi_inputs, pos_x, pos_y, logre, ee1, ee2,
                    fluxes,                  # post-S1 fluxes used for preconditioner
                    neighbor_idx, neighbor_mask,
                    pos_step_max_pix=pos_step_max_pix,
                    damp=1e-3,
                    cg_max_iters=joint_cg_max_iters,
                    cg_tol=joint_cg_tol,
                    use_preconditioner=joint_use_preconditioner,
                    bg_poly_order=bg_poly_order,
                    reg=reg,
                    sersic_n=sersic_n,
                    free_n=free_n,
                    data_override=data_eff,
                )
                # flux_var/bg_var are not refreshed here -- the final
                # stage1_solve at the end of solve_grid_3stage updates
                # them with full-quality variance.
            elif position_solver == "joint_lsqr":
                new_x, new_y, new_logre, new_ee1, new_ee2 = _joint_position_shape_step(
                    multi_inputs, pos_x, pos_y, logre, ee1, ee2, fluxes, bg_coefs,
                    pos_step_max_pix=pos_step_max_pix,
                    damp=1e-3,
                    cg_max_iters=joint_cg_max_iters,
                    cg_tol=joint_cg_tol,
                    use_preconditioner=joint_use_preconditioner,
                    bg_poly_order=bg_poly_order,
                )
            elif has_exp:
                new_x, new_y, new_logre, new_ee1, new_ee2, damp_arr, _s2_chi2 = stage2_source_newton_step(
                    multi_inputs, pos_x, pos_y, logre, ee1, ee2, fluxes, bg_coefs,
                    max_step_pix=pos_step_max_pix,
                    direct_search_step_pix=pos_step_max_pix,
                    direct_search_exp_only=False,
                    damp=damp_arr,
                    bg_poly_order=bg_poly_order,
                    data_override=data_eff,
                )
            else:
                new_x, new_y, damp_arr = stage2_position_newton_step(
                    multi_inputs, pos_x, pos_y, fluxes, bg_coefs,
                    max_step_pix=pos_step_max_pix,
                    damp=damp_arr,
                    bg_poly_order=bg_poly_order,
                    # S2 iteration 1 runs at S1's positions: reuse S1's
                    # rendered stamps (bit-identical). Later iterations
                    # (s2_max_iters>1) run at moved positions -> re-render.
                    precomputed_raw_stamps=s1_stamps if it == 0 else None,
                )
                new_logre, new_ee1, new_ee2 = logre, ee1, ee2
            sec_stage = time.perf_counter() - t_stage
            steps = np.hypot(new_x - pos_x, new_y - pos_y)
            # Free-n rows fold their Sersic-index step into the shape
            # metric (dn == 0 identically for every other row/solver).
            shape_steps = np.sqrt(
                (new_logre - logre) ** 2 + (new_ee1 - ee1) ** 2 + (new_ee2 - ee2) ** 2
                + (new_n - sersic_n) ** 2
            )
            max_step = float(np.max(steps))
            med_step = float(np.median(steps))
            max_shape_step = float(np.max(shape_steps[source_type != SOURCE_STAR])) if has_exp else 0.0
            n_conv = int(np.sum((steps < pos_dlnp_tol_pix) & (shape_steps < pos_dlnp_tol_pix)))
            n_rejected = int(np.sum(steps == 0.0))
            pos_x = new_x
            pos_y = new_y
            logre, ee1, ee2 = new_logre, new_ee1, new_ee2
            sersic_n = np.asarray(new_n, dtype=np.float32)
            if log_fn is not None:
                log_fn(
                    f"S2 it={it+1:02d}  max_step={max_step:8.4f} px  "
                    f"med={med_step:7.4f} px  max_shape={max_shape_step:7.4f}  "
                    f"n_conv={n_conv}/{n_sources}  "
                    f"n_rejected={n_rejected}  tol={pos_dlnp_tol_pix:g}  "
                    f"dt={sec_stage:.2f}s"
                )
            if max(max_step, max_shape_step) < pos_dlnp_tol_pix:
                s2_ok = True
                break

    # Rebuild neighbor list if positions moved significantly.
    if np.max(np.hypot(pos_x - multi_inputs.pos_init_x, pos_y - multi_inputs.pos_init_y)) > 2.0:
        neighbor_idx, neighbor_mask = build_neighbor_list(
            pos_x, pos_y, H_psf, W_psf, k_max=k_max_neighbors,
        )

    # ---- S3 (alternating flux+bg refit + position step) ----
    # s3_max_outer_iters == 0 (the island-mode S1-only passes:
    # single_stage + n_opt_iters: 0) skips the loop entirely; zero
    # iterations were requested, so nothing can be unconverged and the
    # run reports converged with 0 outer iters. (Previously this
    # combination crashed on an undefined max_step in the
    # "did NOT converge" log below.)
    s3_ok = s3_max_outer_iters <= 0
    s3_iters_used = 0
    stranded_mask = np.zeros(n_sources, dtype=bool)
    # Delta-chi2 convergence trackers (extended-source Newton path only).
    chi2_prev = None
    dchi2_prev_small = None
    for it in range(s3_max_outer_iters):
        s3_iters_used = it + 1
        newton_chi2 = None
        # Wing peeling: refresh the per-band (data - wing) images from the
        # CURRENT fluxes/shapes/n (no-op None when peeling is off or the
        # grid has no big sources).
        data_eff = _wing_data_eff(fluxes)
        # Refit fluxes + bg with current positions FIRST. This corrects
        # for any S2 position drift (faint-source flux solve adapts to
        # the new pos), avoiding the chase-noise-pocket pathology that
        # pure position-only iteration suffers from.
        # SKIPPED for joint_varpro -- the varpro step computes the
        # optimal F*, B* INSIDE every chi^2 evaluation, so a separate
        # linear solve here would be redundant work.
        if position_solver == "joint_varpro":
            sec_lin = 0.0
            s3_stamps = None
        else:
            t_lin = time.perf_counter()
            s3_results, s3_lin_ok, s3_stamps = stage1_solve(
                multi_inputs, pos_x, pos_y, logre, ee1, ee2,
                neighbor_idx, neighbor_mask, reg=reg,
                bg_poly_order=bg_poly_order,
                return_stamps=True,
                data_override=data_eff,
            )
            sec_lin = time.perf_counter() - t_lin
            fluxes = np.stack([r.fluxes for r in s3_results], axis=1)
            bg_coefs = np.stack([r.bg_coefs for r in s3_results], axis=0)
            flux_var = np.stack([r.flux_var for r in s3_results], axis=1)
            bg_var = np.stack([r.bg_var for r in s3_results], axis=0)

        # Position step using the NEW flux + bg.
        t_pos = time.perf_counter()
        new_n = sersic_n   # only the varpro branch below updates n
        if position_solver == "joint_varpro":
            new_x, new_y, new_logre, new_ee1, new_ee2, new_n, fluxes, bg_coefs = _joint_varpro_step(
                multi_inputs, pos_x, pos_y, logre, ee1, ee2,
                fluxes,
                neighbor_idx, neighbor_mask,
                pos_step_max_pix=pos_step_max_pix,
                damp=1e-3,
                cg_max_iters=joint_cg_max_iters,
                cg_tol=joint_cg_tol,
                use_preconditioner=joint_use_preconditioner,
                bg_poly_order=bg_poly_order,
                reg=reg,
                sersic_n=sersic_n,
                free_n=free_n,
                data_override=data_eff,
            )
        elif position_solver == "joint_lsqr":
            new_x, new_y, new_logre, new_ee1, new_ee2 = _joint_position_shape_step(
                multi_inputs, pos_x, pos_y, logre, ee1, ee2, fluxes, bg_coefs,
                pos_step_max_pix=pos_step_max_pix,
                damp=1e-3,
                cg_max_iters=joint_cg_max_iters,
                cg_tol=joint_cg_tol,
                use_preconditioner=joint_use_preconditioner,
                bg_poly_order=bg_poly_order,
            )
        elif has_exp:
            new_x, new_y, new_logre, new_ee1, new_ee2, damp_arr, newton_chi2 = stage2_source_newton_step(
                multi_inputs, pos_x, pos_y, logre, ee1, ee2, fluxes, bg_coefs,
                max_step_pix=pos_step_max_pix,
                direct_search_step_pix=max(
                    pos_dlnp_tol_pix,
                    pos_step_max_pix * (0.5 ** it),
                ),
                direct_search_exp_only=False,
                damp=damp_arr,
                bg_poly_order=bg_poly_order,
                data_override=data_eff,
            )
        else:
            new_x, new_y, damp_arr = stage2_position_newton_step(
                multi_inputs, pos_x, pos_y, fluxes, bg_coefs,
                max_step_pix=pos_step_max_pix,
                damp=damp_arr,
                bg_poly_order=bg_poly_order,
                # This iteration's lin refit above rendered at the SAME
                # positions (pos only updates after the step below):
                # reuse its stamp buffers, bit-identical by construction.
                precomputed_raw_stamps=s3_stamps,
            )
            new_logre, new_ee1, new_ee2 = logre, ee1, ee2
        sec_pos = time.perf_counter() - t_pos
        steps = np.hypot(new_x - pos_x, new_y - pos_y)
        # Free-n rows fold their Sersic-index step into the shape metric
        # (dn == 0 identically for every other row/solver).
        shape_steps = np.sqrt(
            (new_logre - logre) ** 2 + (new_ee1 - ee1) ** 2 + (new_ee2 - ee2) ** 2
            + (new_n - sersic_n) ** 2
        )
        max_step = float(np.max(steps))
        med_step = float(np.median(steps))
        max_shape_step = float(np.max(shape_steps[source_type != SOURCE_STAR])) if has_exp else 0.0
        # Per-source convergence (NaN-safe: a NaN step compares False ->
        # counted unconverged, matching the legacy max()-based rule).
        per_source_conv = (steps < pos_dlnp_tol_pix) & (shape_steps < pos_dlnp_tol_pix)
        # dlnp-style secondary criterion (Ceres ftol analogue), scoped to the
        # extended-source Newton path: a source whose position/shape step
        # achieved a chi^2 IMPROVEMENT below chi2_ftol on TWO consecutive
        # iterations has exhausted the data's information even if its
        # parameter steps still ping-pong (oscillators on flat/multimodal
        # chi^2 terrain; rejected steps count as gain 0). Gain is measured
        # within one iteration's flux/neighbor state, so the S1 landscape
        # drift between outer iterations does not pollute it. Star-only
        # grids never produce newton_chi2 -> exact legacy behavior.
        if chi2_ftol > 0 and newton_chi2 is not None:
            gain_small = np.isfinite(newton_chi2) & (newton_chi2 < chi2_ftol)
            if dchi2_prev_small is not None:
                per_source_conv = per_source_conv | (gain_small & dchi2_prev_small)
            dchi2_prev_small = gain_small
        n_conv = int(np.sum(per_source_conv))
        stranded_mask = ~per_source_conv
        # n_rejected: sources whose step was zero (line search rejected
        # all 3 backtrack alphas -> no descent direction found).
        n_rejected = int(np.sum(steps == 0.0))
        pos_x = new_x
        pos_y = new_y
        logre, ee1, ee2 = new_logre, new_ee1, new_ee2
        sersic_n = np.asarray(new_n, dtype=np.float32)
        # Quantile stop: frac=0 reproduces the legacy rule exactly
        # (every source below tol); frac>0 tolerates a tiny stranded
        # tail, reported per source instead of burning whole iterations.
        n_unconv = n_sources - n_conv
        # ceil (not floor) so the knob keeps its documented intent on SMALL
        # grids too: with frac=0.002 and n_sources=220, floor gave 0 allowed
        # stragglers and a single oscillating source burned every remaining
        # iteration (observed: 219/220 converged by it=15, ceiling at 20).
        # SCOPED to grids containing extended sources: star-only grids keep
        # the exact legacy floor rule so pure point-source catalogs remain
        # bit-identical to the pre-extended-source code (regression gate).
        if pos_stop_unconverged_frac > 0 and has_exp:
            n_allowed = int(np.ceil(pos_stop_unconverged_frac * n_sources))
        else:
            n_allowed = int(np.floor(pos_stop_unconverged_frac * n_sources))
        converged = n_unconv <= n_allowed
        if log_fn is not None:
            if converged and n_unconv > 0:
                tag = f"  CONVERGED (quantile stop, stranded={n_unconv})"
            elif converged:
                tag = "  CONVERGED"
            else:
                tag = ""
            log_fn(
                f"S3 it={it+1:02d}/{s3_max_outer_iters}  "
                f"max_step={max_step:8.4f} px  med={med_step:7.4f} px  "
                f"max_shape={max_shape_step:7.4f}  n_conv={n_conv}/{n_sources}  "
                f"n_rejected={n_rejected}  "
                f"tol={pos_dlnp_tol_pix:g}  lin={sec_lin:.2f}s  "
                f"pos={sec_pos:.2f}s{tag}"
            )
        if converged:
            s3_ok = True
            break

    if log_fn is not None and not s3_ok:
        log_fn(
            f"S3 did NOT converge after {s3_iters_used}/{s3_max_outer_iters} iters  "
            f"(last max_step={max_step:8.4f} px > tol={pos_dlnp_tol_pix:g})"
        )

    # Final flux+bg refit with the converged positions to get the
    # output fluxes + variances at the SAME positions.
    # stage1_solve renders from multi_inputs.sersic_n -- publish the
    # (possibly fitted) Sersic-n state so free-n rows refit their flux
    # against the model they converged on (no-op when n never moved).
    multi_inputs.sersic_n = sersic_n
    # Wing peeling: final wing from the converged state.
    data_eff = _wing_data_eff(fluxes)
    t_stage = time.perf_counter()
    if single_stage and s3_max_outer_iters <= 0 and data_eff is None:
        # Island-mode S1-only pass: no position/shape/n parameter moved
        # and there is no wing image, so the final solve would be
        # bit-identical to the init linear solve above -- reuse it
        # (saves one full per-band sparse solve per grid per pass).
        final_results = s1_results
        if log_fn is not None:
            log_fn("final linear solve skipped (0 joint iters; init "
                   "solve reused)")
    else:
        final_results, _final_ok = stage1_solve(
            multi_inputs, pos_x, pos_y, logre, ee1, ee2,
            neighbor_idx, neighbor_mask, reg=reg,
            bg_poly_order=bg_poly_order,
            data_override=data_eff,
        )
        if log_fn is not None:
            log_fn(f"final linear solve done in {time.perf_counter() - t_stage:.2f}s")

    fluxes = np.stack([r.fluxes for r in final_results], axis=1)
    bg_coefs = np.stack([r.bg_coefs for r in final_results], axis=0)
    flux_var = np.stack([r.flux_var for r in final_results], axis=1)
    bg_var = np.stack([r.bg_var for r in final_results], axis=0)

    # ---- Blank-response (empty-aperture) median correction (opt-in) ----
    # The classical deep-photometry bias correction, adapted to the joint
    # solve: unmodeled ambient point-light is positively skewed, so the
    # L2 background sits at its local MEAN while the TYPICAL source
    # aperture contains less ambient than that mean -> every fitted flux
    # carries an additive bias equal to the matched-filter response a
    # PHANTOM source would measure at a typical (blank) position of the
    # FINAL residual. Measure exactly that -- response map = corr(w*r,
    # phi)/corr(w, phi^2) over the whole band, median over good-center
    # pixels -- and subtract it from every flux. Self-consistent with
    # whatever weighting produced the final solve (plain L2 OR
    # confusion-weighted: it removes the residual bias of the estimator
    # that actually ran, deficit and over-correction alike). Derived, no
    # tuned constants; the correction uncertainty (median SE over ~1e5
    # effective apertures) is negligible vs FLUXERR.
    bhat_applied = None
    if blank_response_correction:
        from .error_model import _corr_same
        from .psf_polynomial import evaluate_psf_at
        from .forced_phot_sparse import render_source_stamps, stamp_top_left
        from .source_model import SOURCE_STAR as _BR_STAR
        from .source_model import render_star_stamps as _render_star_stamps

        t_br = time.perf_counter()
        _br_px = jnp.asarray(np.asarray(pos_x, np.float32))
        _br_py = jnp.asarray(np.asarray(pos_y, np.float32))
        _br_st = jnp.asarray(np.asarray(source_type, np.int32))
        _br_ext = np.asarray(source_type) != _BR_STAR
        _br_lr = jnp.asarray(
            np.where(_br_ext, logre, 0.0).astype(np.float32))
        _br_e1 = jnp.asarray(
            np.where(_br_ext, ee1, 1.0).astype(np.float32))
        _br_e2 = jnp.asarray(
            np.where(_br_ext, ee2, 0.0).astype(np.float32))
        _br_sn = jnp.asarray(np.asarray(sersic_n, np.float32))
        _br_flux = np.where(np.isfinite(fluxes), fluxes, 0.0).astype(
            np.float32)
        b_hat = np.zeros(int(multi_inputs.n_bands), dtype=np.float64)
        for _bi in range(int(multi_inputs.n_bands)):
            _inp = multi_inputs.bands[_bi]
            _bh, _bw = int(_inp.band_h), int(_inp.band_w)
            _S = int(_inp.psf_jax.H_psf)
            if _br_ext.any():
                _stamps = render_source_stamps(
                    _inp.psf_jax, _br_px, _br_py, _inp.grid_x0, _inp.grid_y0,
                    _br_st, _br_lr, _br_e1, _br_e2, sersic_n=_br_sn,
                    render_backend=multi_inputs.render_backend)
            else:
                # T6/G2: all-STAR grid -> exact star-only render (bitwise
                # the STAR rows of the full chain; no dead galaxy work).
                _stamps = _render_star_stamps(
                    _inp.psf_jax, _br_px, _br_py, _inp.grid_x0, _inp.grid_y0)
            _stamps = _stamps * jnp.asarray(
                _br_flux[:, _bi])[:, None, None]
            _ixt, _iyt = stamp_top_left(
                _br_px, _br_py, _inp.grid_x0, _inp.grid_y0, _S, _S)
            _model = _scatter_stamps_to_band(
                _stamps, _ixt, _iyt, _bh, _bw)
            if _inp.bg_basis_flat is not None:
                _model = _model + (
                    _inp.bg_basis_flat
                    @ jnp.asarray(final_results[_bi].bg_coefs)
                ).reshape(_bh, _bw)
            _dat = (jnp.asarray(data_eff[_bi]) if data_eff is not None
                    else _inp.data)
            _iv = _inp.invvar
            _good = _inp.pix_good & (_iv > 0)
            _w = jnp.where(_good, _iv, 0.0)
            _r = jnp.where(_good, _dat - _model, 0.0)
            _phi = evaluate_psf_at(
                _inp.psf_jax,
                jnp.float32(_inp.grid_x0 + 0.5 * _bw),
                jnp.float32(_inp.grid_y0 + 0.5 * _bh))
            _phi = jnp.clip(_phi, 0.0, None)
            _phi = _phi / jnp.sum(_phi)
            _num = _corr_same(_w * _r, _phi)
            _den = _corr_same(_w, _phi * _phi)
            _resp = jnp.where(_den > 0, _num / jnp.maximum(_den, 1e-30),
                              jnp.nan)
            _resp = jnp.where(_good, _resp, jnp.nan)
            b_hat[_bi] = float(jnp.nanmedian(_resp))
        # STAR rows only: the response was measured with the PSF template,
        # so it is exactly the bias of a PSF-photometered source. Extended
        # sources have a larger matched aperture and measurably different
        # net systematics -- applying the star-template correction to them
        # moves their (already bright-side) bias the WRONG way (oracle:
        # snr30 galaxies -6 -> -30 mmag). Same lesson as the all-types ->
        # STAR-only scene-subtract refinement.
        _br_star = ~_br_ext
        fluxes[_br_star, :] = (
            fluxes[_br_star, :] - b_hat[None, :].astype(fluxes.dtype))
        bhat_applied = b_hat.copy()
        if log_fn is not None:
            log_fn(
                "blank-response correction done in "
                f"{time.perf_counter() - t_br:.2f}s (per-band median "
                f"response: min {b_hat.min():+.4g} / "
                f"max {b_hat.max():+.4g} flux units, subtracted from "
                f"{int(_br_star.sum())} STAR flux rows"
                + "; b_hat per band: "
                + np.array2string(b_hat, precision=4,
                                  max_line_width=10 ** 6) + ")")

    # Per-source Sersic-n uncertainty at the converged point (free-n rows
    # only; diagonal / flux-marginalized -- see _varpro_sersic_n_err).
    sersic_n_err = np.full(n_sources, np.nan, dtype=np.float32)
    if position_solver == "joint_varpro" and bool(np.any(free_n)):
        t_stage = time.perf_counter()
        sersic_n_err = _varpro_sersic_n_err(
            multi_inputs, pos_x, pos_y, logre, ee1, ee2,
            sersic_n, free_n, neighbor_idx, neighbor_mask,
            bg_poly_order=bg_poly_order, reg=reg,
            data_override=data_eff,
        )
        if log_fn is not None:
            log_fn(
                f"sersic_n_err for {int(np.sum(free_n))} free-n source(s) "
                f"in {time.perf_counter() - t_stage:.2f}s"
            )

    # Position variance: per-source 2x2 Hessian inverse, diagonal.
    # For now, approximate via 1/(prior_inv_sigma^2 + small flux-related term).
    # Skipped for simplicity in v1; populated as inv(prior_inv_sigma**2).
    pos_var_xy = 1.0 / np.maximum(
        np.asarray(multi_inputs.pos_err, dtype=np.float32) ** 2 +
        1.0 / np.maximum(np.sum(fluxes, axis=1) ** 2, 1e-30),
        1e-30,
    )
    pos_var = np.stack([pos_var_xy, pos_var_xy], axis=1)  # (n_sources, 2)

    # shape_frozen (task: seed->output honesty): the per_source / joint_lsqr
    # position steps thaw shape ONLY for EXP (is_exp gate), so a DEV/SERSIC
    # row fit under those solvers keeps its Re/ELL/THETA at the catalog seed
    # -- reported as *_fit with no other flag. Emit an informational flag so
    # the seed-echoed shape is self-evident. joint_varpro fits DEV/SERSIC
    # shapes (K=6 keep mask), so it is honestly False there; island mode
    # runs per_source per-grid but the island stage refits shapes, so the
    # pipeline OVERRIDES this to False for island runs (mirrors the
    # sersic_n_frozen island override).
    is_dev_or_sersic = (source_type == SOURCE_DEV) | (source_type == SOURCE_SERSIC)
    if position_solver in ("per_source", "joint_lsqr"):
        shape_frozen = np.asarray(is_dev_or_sersic, dtype=bool)
    else:
        shape_frozen = np.zeros(n_sources, dtype=bool)

    return GridMultiBandResult(
        pos_x=pos_x, pos_y=pos_y,
        logre=logre, ee1=ee1, ee2=ee2,
        fluxes=fluxes, bg_coefs=bg_coefs,
        flux_var=flux_var, bg_var=bg_var,
        pos_var=pos_var,
        n_outer_iters=s3_iters_used,
        s1_converged=s1_ok, s2_converged=s2_ok, s3_converged=s3_ok,
        position_stranded=stranded_mask,
        sersic_n_fit=sersic_n.astype(np.float32),
        # DEV rows report a structurally-fixed n=4 that is never fit, so it
        # is frozen too (free_n is SERSIC-only, so DEV -> True); this makes
        # the reported DEV sersic_n_fit honestly self-describe as a seed.
        sersic_n_frozen=(
            ((source_type == SOURCE_SERSIC) | (source_type == SOURCE_DEV))
            & ~free_n),
        shape_frozen=shape_frozen,
        sersic_n_err=sersic_n_err,
        blank_response_bhat=bhat_applied,
    )


__all__ = [
    "GridMultiBandInputs",
    "GridMultiBandResult",
    "stage1_solve",
    "stage2_position_newton_step",
    "stage2_source_newton_step",
    "solve_grid_3stage",
]
