"""Island-fit class planning + host-side array prep (island-arch #25).

Pure host-side NumPy helpers shared by the single-process solver
(:mod:`.island_solver`) and the 4-GPU sharded island stage
(:mod:`.island_stage`). Deliberately import-light: NO JAX and no
grid_gpu modules that pull JAX at import time (mirrors
:mod:`.islands`), so the PIPELINE PARENT can plan the work, build the
shared chunk queue, and unpack per-class results without paying the
JAX import or touching the GPU.

Concepts (design the internal design note ``DESIGN_island_arch.md`` (not distributed)
sections 6b + S5):

- A CLASS is one compiled-program family: a ``(bbox_class B,
  nsrc_class N, kind star|mixed, stamp tier T)`` cell -- exactly the
  sub-bucket unit the solver already iterated. :func:`plan_island_classes`
  reproduces the bucket -> tier-group split, the S5 ``joint_nsrc_cap``
  giant capping, and the peak-memory chunk-width model of the P3b/P5
  solver, and returns one :class:`IslandClassPlan` per occupied class.
  The vmap CHUNK WIDTH is no longer decided here (T5/#28): it is measured
  on the device by ``island_solver.measure_class_chunk_width``.
- A CHUNK (the measured number of islands of one class, padded) is the
  SCHEDULING UNIT for the multi-GPU work queue (ratified spec 6b):
  islands within a bucket are interchangeable, so chunks of one class
  can run on different GPUs.
- The DIFFICULTY RE-QUEUE (S5 "batch cost = slowest member" law) runs
  every chunk for at most ``max_steps_main`` LM steps; islands that
  neither converged nor stalled are re-queued into compact "hard"
  chunks a quarter as wide (same kernel family, narrower
  batch) for the remaining step budget, carrying their exact LM state
  (p, lambda, chi2, consecutive-ftol / reject counters, step count) --
  the continued trajectory is mathematically identical to the
  single-pass solver; only the vmap batch width changes (ulp-level fp
  reassociation, the documented chunk-width noise class).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

# Matches tract7dt.fit.grid_gpu.source_model constants; duplicated here
# (plain ints/floats) to keep this module importable without JAX
# (source_model imports the JAX lanczos kernel at module level).
SOURCE_STAR = 0
SOURCE_SERSIC = 3
SERSIC_N_MIN = 0.29
SERSIC_N_MAX = 6.3

# LM / convergence defaults shared by the solver and the (JAX-free)
# stage parent. Full rationale in island_solver.py's docstring/comments
# (P3b convergence policy); values mirror production knobs.
DEFAULT_MAX_STEPS = 30
# #19a Re profile-likelihood scan offsets (natural-log Re), symmetric;
# +-2.5 ln = x12 each way. K=12.
REPROF_DELTAS_LN = np.array(
    [-2.5, -1.5, -0.9, -0.5, -0.25, -0.1,
     0.1, 0.25, 0.5, 0.9, 1.5, 2.5], dtype=np.float32)

DEFAULT_CHI2_FTOL_REL = 1e-4     # P3 policy: dchi2 < ftol_rel*max(chi2,1)
DEFAULT_STEP_TOL_PIX = 0.1       # production dlnp_stop_pix
DEFAULT_POS_STEP_MAX = 0.5       # production solver.pos_step_max_pix
DEFAULT_UN_FREEZE_STEPS = 0      # u_n annealing budget (0 = never; opt-in)

# Difficulty re-queue defaults (S5): most islands converge in 2-7 steps
# (convexp evidence); 8 main steps keep them from paying for 30-step
# stragglers, which finish in narrow hard chunks.
DEFAULT_MAX_STEPS_MAIN = 8
# Hard chunks are ``chunk_main / HARD_CHUNK_DIV`` wide (floor 1): narrow
# enough that a straggler-among-stragglers only holds a few islands
# hostage, wide enough to keep some vmap parallelism.
HARD_CHUNK_DIV = 4

# ---------------------------------------------------------------------------
# REMOVED 2026-08-02 (T5 / roadmap #28): the hand-calibrated peak-memory
# model (_MEM_A_J ... _MEM_SCAN_DB) and its hand-set budget
# TRACT7DT_ISLAND_STEP_MEM_GB. It priced the chunk width as
# `per-island bytes x chunk` against a fixed GB ceiling, and all three
# parts were measured wrong on 2026-08-02 (development ledger 6-4):
#   - no fixed-cost term, so a small-chunk calibration folded the fixed
#     cost into the slope and over-charged up to 6x at wide chunks;
#   - it was calibrated through island_fit, which cannot even run the
#     marginal-flux kernel production runs (that kernel needs 4x what the
#     LM step needs at nsrc 32 -- measured);
#   - the marginal-flux jacrev intermediate is `width x N^2 x B^2`, a
#     product of a bucket property and the chunk, which no per-island
#     constant can express.
# It is replaced by island_solver.measure_class_chunk_width: budget = the
# card's free memory, cost = what XLA reports for every kernel that will
# actually run. Nothing here models memory any more.
# ---------------------------------------------------------------------------
# debug/probe overrides (used by analysis/island_p3b_memprobe.py)
_CHUNK_OVERRIDE = os.environ.get("TRACT7DT_ISLAND_CHUNK")
_TC_OVERRIDE = os.environ.get("TRACT7DT_ISLAND_TANGENT_CHUNK")

# Relative per-island-step COMPUTE cost factor of a mixed (fft-render)
# lane vs a star lane at equal working-set bytes. Rough by design (the
# out_injectlight per-bucket walls put it at ~2.5-4x); the LPT queue
# only needs the right ORDERING and the dynamic worker pulls absorb
# the residual error (spec 6b).
_COST_KIND_FACTOR = {"star": 1.0, "mixed": 4.0}


def ladder_width(n_rows: int, cap: int) -> int:
    """Smallest power-of-two vmap width that holds ``n_rows`` (<= ``cap``).

    T5/#28: chunk widths are ladder rungs (see
    ``island_solver.measure_class_chunk_width``), so a PARTIAL chunk pads
    out to its own rung rather than to the full class width -- a 5-island
    tail of a 4096-wide class runs 8 wide, not 4096. Every width a run uses
    is therefore one of ~13 shapes, which is also what lets the persistent
    JIT cache carry compiled kernels from one field to the next instead of
    every field compiling its own arbitrary widths.
    """
    n = max(1, int(n_rows))
    w = 1 << (n - 1).bit_length()
    return int(min(max(w, 1), max(1, int(cap))))


def sersic_n_to_u_np(n: np.ndarray) -> np.ndarray:
    """NumPy twin of source_model.sersic_n_to_u (logit bijection)."""
    t = (np.clip(n, SERSIC_N_MIN + 1e-4, SERSIC_N_MAX - 1e-4)
         - SERSIC_N_MIN) / (SERSIC_N_MAX - SERSIC_N_MIN)
    return np.log(t / (1.0 - t))


def _propagate_shape_errors_np(
    ee1: np.ndarray, ee2: np.ndarray, re: np.ndarray,
    ell_err: np.ndarray, theta_err: np.ndarray, re_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized JAX-free port of ``fit/runner._propagate_shape_errors``.

    Propagates catalog-frame prior sigmas (``Re_PRIOR_SIG``,
    ``ELL_PRIOR_SIG``, ``THETA_PRIOR_SIG``) to the internal
    ``(logre, ee1, ee2)`` prior sigmas using the analytic Jacobian of
    ``EllipseESoft.rAbPhiToESoft``. The catalog ``(ell, phi)`` are
    re-derived from the internal seed ``(ee1, ee2)`` (the exact inverse
    of ``ellipse_soft_from_re_ab_phi_np`` -- ``cos/sin(2*phi)`` and the
    ``ee`` magnitude are recovered bit-for-bit up to clipping), so the
    Jacobian matches the CPU path evaluated at the same seed.

    Inputs are per-member arrays (NaN in a sigma column means "no
    prior"); the same value-frame and the same ``both-or-neither`` rule
    on ``ELL_PRIOR_SIG``+``THETA_PRIOR_SIG`` as the CPU version.
    Returns ``(sigma_logre, sigma_ee1, sigma_ee2)`` with NaN where the
    corresponding prior is absent/invalid.
    """
    ee1 = np.asarray(ee1, dtype=np.float64)
    ee2 = np.asarray(ee2, dtype=np.float64)
    re = np.asarray(re, dtype=np.float64)
    ell_err = np.asarray(ell_err, dtype=np.float64)
    theta_err = np.asarray(theta_err, dtype=np.float64)
    re_err = np.asarray(re_err, dtype=np.float64)

    # sigma_logre = Re_PRIOR_SIG / Re   (exp(logre) frame, no q coupling)
    re_ok = np.isfinite(re_err) & (re_err > 0) & (re > 0)
    sigma_logre = np.where(
        re_ok, re_err / np.where(re > 0, re, 1.0), np.nan)

    # catalog (ell, phi) from the internal seed (inverse of the soft
    # ellipse map); only cos/sin(2*phi) and the ee magnitude matter.
    ee_mag_seed = np.hypot(ee1, ee2)
    e_seed = 1.0 - np.exp(-ee_mag_seed)
    ab = (1.0 - e_seed) / np.maximum(1.0 + e_seed, 1e-12)
    phi = -np.degrees(np.arctan2(ee2, ee1) / 2.0)
    ell = 1.0 - ab

    ell_c = np.clip(ell, 0.0, 0.999)
    angle = np.radians(-2.0 * phi)
    ba = 1.0 - ell_c
    e = (1.0 - ba) / (1.0 + ba)
    ee = -np.log(np.maximum(1.0 - e, 1e-15))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    dee_dell = 1.0 / np.maximum((2.0 - ell_c) * (1.0 - ell_c), 1e-15)
    dee1_dell = dee_dell * cos_a
    dee2_dell = dee_dell * sin_a
    k = 2.0 * np.pi / 180.0
    dee1_dtheta = k * ee * sin_a
    dee2_dtheta = -k * ee * cos_a

    ee_ok = (np.isfinite(ell_err) & (ell_err > 0)
             & np.isfinite(theta_err) & (theta_err > 0))
    # NaN-safe products: masked out via ee_ok below.
    ell_e = np.where(ee_ok, ell_err, 0.0)
    th_e = np.where(ee_ok, theta_err, 0.0)
    ee1_var = (dee1_dell * ell_e) ** 2 + (dee1_dtheta * th_e) ** 2
    ee2_var = (dee2_dell * ell_e) ** 2 + (dee2_dtheta * th_e) ** 2
    sigma_ee1 = np.where(ee_ok & (ee1_var > 0),
                         np.sqrt(np.where(ee1_var > 0, ee1_var, 1.0)), np.nan)
    sigma_ee2 = np.where(ee_ok & (ee2_var > 0),
                         np.sqrt(np.where(ee2_var > 0, ee2_var, 1.0)), np.nan)
    return sigma_logre, sigma_ee1, sigma_ee2


def free_n_gate(
    source_type: np.ndarray,
    s1_flux: np.ndarray,
    s1_flux_var: np.ndarray,
    fit_sersic_n: bool,
    n_fit_min_snr: float,
) -> np.ndarray:
    """Production free-n S/N gate from the S1 fluxes (per source)."""
    n_src = int(source_type.shape[0])
    free_n_all = np.zeros(n_src, dtype=bool)
    if fit_sersic_n and bool(np.any(source_type == SOURCE_SERSIC)):
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_b = np.abs(s1_flux) / np.sqrt(np.maximum(s1_flux_var, 1e-30))
        snr = np.nanmedian(snr_b, axis=1)
        free_n_all = (
            (source_type == SOURCE_SERSIC)
            & np.isfinite(snr) & (snr >= float(n_fit_min_snr))
        )
    return free_n_all


@dataclass
class IslandFitResult:
    """PER-SOURCE results for every source in a bucketed or giant island.

    Arrays are aligned with ``source_idx`` (indices into the caller's
    source arrays). Island-level quantities (chi2, n_steps, converged,
    giant flag) are replicated onto each member row. ``converged=False``
    rows form the re-queue list. (Lives here -- not in island_solver --
    so the JAX-free pipeline parent can assemble sharded results.)
    """
    bandnames: list
    source_idx: np.ndarray          # (n_fit,) int64
    pos_x: np.ndarray               # (n_fit,)
    pos_y: np.ndarray
    logre: np.ndarray
    ee1: np.ndarray
    ee2: np.ndarray
    sersic_n: np.ndarray
    fluxes: np.ndarray              # (n_fit, n_bands)
    flux_err: np.ndarray            # (n_fit, n_bands) sqrt(diag inv(A))
    chi2: np.ndarray                # (n_fit,) island chi2
    n_steps: np.ndarray             # (n_fit,) int32
    converged: np.ndarray           # (n_fit,) bool
    free_n: np.ndarray              # (n_fit,) bool
    stalled: np.ndarray = None      # (n_fit,) bool -- converged via prong 3
    # #35 stall-anatomy disclosure (replicated per member): the island's LM
    # trajectory contained STALL_REJECT_K consecutive rejected steps whose
    # clip/bound-limited proposal was IDENTICAL each time (the lam escalation
    # never tested a distinct step -- the clip-saturated-runaway class).
    # Under the default stall_count_mode="proposals" this is a subset of
    # ``stalled``; under "changed" it marks islands that continued past the
    # frozen-proposal episode. False for giants (never LM-stepped).
    diverged: np.ndarray = None     # (n_fit,) bool
    # Stall-yield (2026-07-27): probability-convergence certificate for
    # stalled islands -- best available damped-step chi2 improvement over
    # the full lambda ladder (negative = improvement exists; near-zero =
    # converged-on-a-plateau). NaN for non-stalled rows and giants.
    stall_dchi2: np.ndarray = None   # (n_fit,) float32
    railed: np.ndarray = None        # (n_fit,) bool (trust-region bound-pinned)
    # LEG D nonlinear (task #20): per-island reduced chi2 = chi2 / dof, dof =
    # n_good_pixels - n_free_params (kept nonlinear + n_bands*active members),
    # from the SAME sky-marginal whitened residual as the fit. PER-ISLAND
    # (shared by members -- the residual is joint). >=0; NaN where undefined
    # (giants, dof<=0). Feeds the nonlinear *_err_tot columns (C x D).
    chi2_nu: np.ndarray = None      # (n_fit,)
    island_id: np.ndarray = None    # (n_fit,) int32
    island_nsrc: np.ndarray = None  # (n_fit,) int32
    island_giant: np.ndarray = None  # (n_fit,) bool
    # (n_fit,) px: per-source fitted-Re upper bound = 0.75 * stamp TIER
    # (tier chosen from the seed shape). NaN for giants (shape frozen).
    # Lets the pipeline make flag_shape_bound tier-aware.
    re_upper_pix: np.ndarray = None
    # ---- #28a post-convergence 1-sigma parameter errors (catalog
    # frame; marginal over the island's kept nonlinear params + fluxes,
    # background frozen). NaN where undefined (stars' shapes, frozen n,
    # giants, degenerate Hessian) or when the knob is off. ----
    pos_x_err: np.ndarray = None    # (n_fit,) px
    pos_y_err: np.ndarray = None    # (n_fit,) px
    re_err: np.ndarray = None       # (n_fit,) px (on Re = exp(logre))
    # #19a Re profile-likelihood interval: the Delta-chi2 <= 1 bounds of
    # the flux-profiled chi2 along log-Re (per-band fluxes re-solved at
    # each scan point). Honest Re width along the flux-Re degeneracy
    # (the local-Fisher re_err misses curved valleys). Clamped at the
    # +-2.5 ln (x12) scan edge = "at least this wide". NaN: stars,
    # giants, knob off.
    re_prof_lo: np.ndarray = None   # (n_fit,) px
    re_prof_hi: np.ndarray = None   # (n_fit,) px
    ell_err: np.ndarray = None      # (n_fit,) dimensionless (on ELL = 1-ab)
    theta_err: np.ndarray = None    # (n_fit,) deg (on THETA)
    sersic_n_err: np.ndarray = None  # (n_fit,) free-n rows only
    # ---- R1 prior_frac disclosure: per prior-carrying fitted param, the
    # fraction of its joint-fit constraint from the catalog prior,
    # prior_inv2/(H_kk+prior_inv2) in [0,1] (0 pure data .. 1 pure prior).
    # NaN where the param carries no prior / is not fitted / giant. ----
    pos_x_prior_frac: np.ndarray = None    # (n_fit,)
    pos_y_prior_frac: np.ndarray = None    # (n_fit,)
    re_prior_frac: np.ndarray = None       # (n_fit,)
    ell_prior_frac: np.ndarray = None      # (n_fit,)
    theta_prior_frac: np.ndarray = None    # (n_fit,)
    sersic_n_prior_frac: np.ndarray = None  # (n_fit,)
    # ---- #17 per-SOURCE model-quality metrics (legacypipe DR9 convention),
    # scored on the converged fit. NaN where undefined (giants, off-island) or
    # when source_metrics_enabled is off. Defined for ALL fitted members incl.
    # stars/frozen-n (they are flux/footprint metrics, not shape errors). ----
    rchisq: np.ndarray = None       # (n_fit,) own-profile-weighted reduced chi2
    fracflux: np.ndarray = None     # (n_fit,) neighbour-flux contamination frac
    fracin: np.ndarray = None       # (n_fit,) analytic-total flux in footprint
    fracmasked: np.ndarray = None   # (n_fit,) model flux on masked pixels
    # #35 Option A disclosure: per-band DATA-CONSTRAINED window flux
    # (island VarPro amplitude x unit-model window sum -- the per-band
    # quantity behind fracin). NaN for giants / when source_metrics is off.
    flux_win: np.ndarray = None     # (n_fit, n_bands)
    # D2: per-band SHAPE-marginalized flux-error INCREMENT variance
    # (n_fit, n_bands). 0 for star rows; NaN for giants / marginal-flux off.
    flux_marg_var: np.ndarray = None
    bucket_stats: list = field(default_factory=list)

    @property
    def requeue_idx(self) -> np.ndarray:
        return self.source_idx[~self.converged]


@dataclass
class IslandClassPlan:
    """One compiled-program class = (bbox B, nsrc N, kind, tier T) cell."""

    class_idx: int                 # position in the plan list
    bbox_class: int                # B
    nsrc_class: int                # N
    kind: str                      # "star" | "mixed"
    tier: int                      # member stamp tier T
    isl_ids: np.ndarray = field(repr=False)      # (n_isl,) island ids
    midx: np.ndarray = field(repr=False)         # (n_isl, N) member idx >= 0
    mmask: np.ndarray = field(repr=False)        # (n_isl, N) bool
    bbox: np.ndarray = field(repr=False)         # (n_isl, 4) island bboxes
    n_src_of: np.ndarray = field(repr=False)     # (n_isl,) true member count
    # PERF #14 (star/galaxy render split): the MAX galaxy-member count
    # (is_ext & mmask) over the class's islands. Static class identity ->
    # the FFT-render sub-batch width G in the mixed kernel (star classes
    # keep 0; they never touch the split). Baked into the kernel-cache key.
    n_gal_class: int = 0
    tangent_chunk: int = 6
    chunk_main: int = 1            # phase-A vmap width (memory model)
    chunk_hard: int = 1            # re-queue vmap width (narrower)
    per_island_bytes: int = 0
    cost_per_island_step: float = 0.0   # LPT cost unit (bytes x kind factor)

    @property
    def n_islands(self) -> int:
        return int(self.isl_ids.shape[0])

    @property
    def key(self) -> tuple:
        return (self.bbox_class, self.nsrc_class, self.kind, self.tier)


def plan_island_classes(
    island_set,
    buckets,
    source_type: np.ndarray,
    logre: np.ndarray,
    sersic_n: np.ndarray,
    native: int,
    *,
    joint_nsrc_cap: int | None = 32,
    nsrc_max: int | None = None,
    log_fn=None,
    crossband_rank: int = 0,
) -> tuple[list, list]:
    """Split the occupied buckets into solver classes + chunk widths.

    Returns ``(plans, capped_giant_ids)``: star/mixed grouping, the
    per-member stamp-tier ladder and the S5 ``joint_nsrc_cap`` giant
    capping. Host-only NumPy.

    T5/#28 (2026-08-02): the chunk width and the memory-infeasible ->
    giant-path downgrade are NO LONGER decided here. Both were
    functions of a hand-calibrated byte model that has been deleted; the
    width is now measured on the device
    (``island_solver.measure_class_chunk_width``) and a class that does not
    fit even one island at a time is downgraded there, on evidence.
    ``crossband_rank`` is accepted for call compatibility and ignored: the
    crossband kernels are what the measurement compiles, so their extra
    working set is measured rather than modelled.
    """
    def log(msg):
        if log_fn is not None:
            log_fn(msg)

    todo = {k: b for k, b in buckets.buckets.items()
            if nsrc_max is None or k[1] <= int(nsrc_max)}
    capped_giant_ids: list = []
    if joint_nsrc_cap is not None:
        over = [k for k in list(todo) if k[1] > int(joint_nsrc_cap)]
        for k in over:
            capped_giant_ids.extend(todo.pop(k).island_ids.tolist())
        if capped_giant_ids:
            log(f"joint_nsrc_cap={joint_nsrc_cap}: {len(capped_giant_ids)} "
                f"island(s) from nsrc buckets {sorted({k[1] for k in over})} "
                f"-> giant path (seed geometry, no island fit)")
    if not todo:
        raise ValueError("no buckets selected")

    plans: list[IslandClassPlan] = []
    for (bc, nc) in sorted(todo):
        b = todo[(bc, nc)]
        ids = b.island_ids
        N = int(nc)
        mem_idx = b.member_idx.astype(np.int64)          # (n_isl, N), -1 pad
        mem_mask = b.member_mask.astype(bool)
        safe_idx = np.maximum(mem_idx, 0)
        # star-only vs mixed islands (any extended member -> mixed);
        # mixed islands additionally sub-bucket by STAMP TIER.
        #
        # TIER = per-MEMBER model support, NOT the bbox size: a
        # multi-star island in a large union bbox still only needs
        # native stamps per member (P3 fix). Mixed islands take the
        # smallest production stamp tier ({55,111,221} ladder) covering
        # 2*nre(n)*Re of their largest extended member.
        is_gal_all = (source_type[safe_idx] != SOURCE_STAR) & mem_mask
        island_has_ext = np.any(is_gal_all, axis=1)
        # PERF #14: per-island galaxy-member count (feeds n_gal_class = G).
        n_gal_per_isl = is_gal_all.sum(axis=1)
        nre_all = np.where(sersic_n[safe_idx] < 1.5, 4.0,
                           np.where(sersic_n[safe_idx] < 3.0, 6.0, 8.0))
        req_half = np.where(
            (source_type[safe_idx] != SOURCE_STAR) & mem_mask,
            nre_all * np.exp(logre[safe_idx].astype(float)), 0.0
        ).max(axis=1)
        tier_ladder = [t for t in (native, 111, 221) if t >= native]
        isl_tier = np.full(ids.size, tier_ladder[-1], dtype=np.int64)
        for t in reversed(tier_ladder):
            isl_tier = np.where(2 * req_half <= t, t, isl_tier)

        groups = [("star", native, np.where(~island_has_ext)[0])]
        for t in tier_ladder:
            groups.append((
                "mixed", int(t),
                np.where(island_has_ext & (isl_tier == t))[0]))

        for kind, T, sel in groups:
            if sel.size == 0:
                continue
            nP = 6 * N
            tangent_chunk = 6
            if _TC_OVERRIDE is not None:
                tangent_chunk = int(_TC_OVERRIDE)
                if nP % tangent_chunk:
                    raise ValueError(
                        f"TRACT7DT_ISLAND_TANGENT_CHUNK={tangent_chunk} "
                        f"does not divide nP={nP}")
            # T5/#28: the chunk width is NOT decided here any more. This
            # module is JAX-free and runs in the pipeline parent, which has
            # neither the compiled kernels nor a device; the width now comes
            # from island_solver.measure_class_chunk_width, run inside the
            # worker that owns the card (XLA's own per-kernel demand vs that
            # card's free memory). What used to sit here -- eight hand-fitted
            # coefficients times a hand-set GB budget -- was measured wrong
            # in three independent ways on 2026-08-02 (no fixed-cost term, a
            # calibration path that could not run the marginal-flux kernel,
            # and a width x N^2 x B^2 term no per-island constant can
            # express). ``chunk_main``/``chunk_hard`` stay as an INERT
            # placeholder so a plan pickled by an older parent still loads;
            # every consumer takes the measured width.
            chunk = int(min(sel.size, 1))
            if _CHUNK_OVERRIDE is not None:
                chunk = int(_CHUNK_OVERRIDE)
            chunk_hard = max(1, chunk // HARD_CHUNK_DIV)
            per_island = 0
            # PERF #14: G = max galaxy count over this class's islands
            # (0 for the star sub-class -- it takes the cheap native path
            # and never renders the FFT). Static -> kernel-cache key.
            n_gal_class = (int(n_gal_per_isl[sel].max())
                           if kind == "mixed" else 0)
            plans.append(IslandClassPlan(
                class_idx=len(plans),
                bbox_class=int(bc), nsrc_class=N, kind=kind, tier=int(T),
                isl_ids=ids[sel].astype(np.int64),
                midx=safe_idx[sel],
                mmask=mem_mask[sel],
                bbox=island_set.bbox[ids[sel]],
                n_src_of=island_set.n_src[ids[sel]].astype(np.int32),
                n_gal_class=n_gal_class,
                tangent_chunk=tangent_chunk,
                chunk_main=chunk,
                chunk_hard=chunk_hard,
                per_island_bytes=per_island,
                cost_per_island_step=(
                    per_island * _COST_KIND_FACTOR.get(kind, 1.0)),
            ))
    return plans, capped_giant_ids


def class_host_arrays(
    plan: IslandClassPlan,
    *,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    source_type: np.ndarray,
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
    sersic_n: np.ndarray,
    free_n_all: np.ndarray,
    pos_max_shift_pix: float,
    pad: int,
    pos_err: np.ndarray | None = None,
    re_prior_sig: np.ndarray | None = None,
    ell_prior_sig: np.ndarray | None = None,
    theta_prior_sig: np.ndarray | None = None,
    sersic_n_prior_sig: np.ndarray | None = None,
    s1_flux: np.ndarray | None = None,
    cb_scale: np.ndarray | None = None,
) -> dict:
    """Per-class NumPy seed/window arrays for the chunk runner + unpack.

    ``pad`` is the border width of the padded field stacks (window
    origins ``ox/oy`` are already offset by it). Pure NumPy; identical
    results in the parent (unpack) and the GPU workers (fit).

    Catalog-driven NONLINEAR Gaussian PRIORS (opt-in; a prior exists ONLY
    where the user supplied its WIDTH sigma column -- input-catalog values
    are SEEDS, not truth). Every ``*_PRIOR_SIG`` per-source array carries
    NaN where the prior is absent/invalid; the propagation below maps that
    to ``inv_sigma = 0`` (the bit-identity sentinel: an all-zero prior is
    an exact no-op). The prior MEAN is the seed itself (``p0``):

    - ``pos_err`` (px) -> Gaussian prior on ``x, y`` (same sigma both
      axes), mean = seed x, y. Any real member with a valid ``POS_ERR``.
    - ``re/ell/theta_prior_sig`` -> ``(logre, ee1, ee2)`` priors via
      :func:`_propagate_shape_errors_np` (exact CPU Jacobian; ee needs
      BOTH ell and theta sigmas). EXTENDED sources only.
    - ``sersic_n_prior_sig`` -> prior on the internal ``u_n`` via the
      logit Jacobian ``inv_sigma_u = (dn/du) / SERSIC_N_PRIOR_SIG`` at
      the seed n; SERSIC rows only. A finite value THAWS the free-n gate
      for that row (``fn_flag |= has_n_prior``), so a strong n-prior
      replaces the S/N-based hard freeze.

    Per-band FLUX priors are DEFERRED to the future field-level flux
    solve (P7 de-gridding): the reported flux comes from the data-only
    pass-2 S1' per-grid solve, so a prior on the discarded internal
    island flux would not affect any reported value.

    ``has_priors`` (returned) is True iff ANY nonlinear inv_sigma in the
    class is nonzero; the solver bakes it into a static kernel-cache flag
    so the no-prior program is byte-identical to the pre-prior kernel.
    """
    B = int(plan.bbox_class)
    N = int(plan.nsrc_class)
    midx = plan.midx
    mmask = plan.mmask
    n_isl = plan.n_islands
    T = int(plan.tier)

    bbox = plan.bbox
    w_box = bbox[:, 2] - bbox[:, 0]
    h_box = bbox[:, 3] - bbox[:, 1]
    ox = bbox[:, 0] - (B - w_box) // 2 + pad
    oy = bbox[:, 1] - (B - h_box) // 2 + pad
    x0b = (ox - pad).astype(np.float32)
    y0b = (oy - pad).astype(np.float32)

    sx = pos_x[midx].astype(np.float32)
    sy = pos_y[midx].astype(np.float32)
    st = source_type[midx].astype(np.int32)
    lr = logre[midx].astype(np.float32)
    e1 = ee1[midx].astype(np.float32)
    e2 = ee2[midx].astype(np.float32)
    sn = sersic_n[midx].astype(np.float32)
    u_seed = sersic_n_to_u_np(sn).astype(np.float32)
    is_ext = st != SOURCE_STAR

    # ---- PERF #14 star/galaxy render split: galaxy-lane gather map ----
    # For each island, gsel (n_isl, G) holds the member-slot indices to
    # render via the table-MoG FFT (the galaxies), gvalid (n_isl, G) marks
    # the real ones (the rest are padding). ``order`` sorts each island's
    # slots GALAXIES-FIRST (stable, so original slot order is preserved),
    # so gsel[:, :G] are DISTINCT slot indices per island and gvalid picks
    # the true galaxies. Because the padding entries land on NON-galaxy
    # slots (distinct from every real galaxy slot), the kernel's
    # scatter-set has unique indices per island and is deterministic; the
    # padding write is a no-op (writes the cheap PSF render back to a
    # non-galaxy slot). Members are NEVER reordered -- gsel is a gather
    # index, the per-member arrays stay in island-member order. Star
    # classes carry G=0 (empty gsel/gvalid); they never take the split.
    # Always built (cheap host NumPy); the kernel ignores them when the
    # star_render_split flag is off.
    G = int(plan.n_gal_class)
    is_gal = is_ext & mmask                          # (n_isl, N) bool
    if G > 0:
        order = np.argsort(~is_gal, axis=1, kind="stable")   # galaxies first
        gsel = order[:, :G].astype(np.int32)                 # (n_isl, G)
        gvalid = np.take_along_axis(is_gal, order, axis=1)[:, :G]
    else:
        gsel = np.zeros((n_isl, 0), dtype=np.int32)
        gvalid = np.zeros((n_isl, 0), dtype=bool)

    # ---- per-member prior sigmas (NaN sentinel = no prior) ----
    def _gather(arr):
        if arr is None:
            return np.full((n_isl, N), np.nan, dtype=np.float64)
        return np.asarray(arr, dtype=np.float64)[midx]

    pe_m = _gather(pos_err)
    re_ps = _gather(re_prior_sig)
    ell_ps = _gather(ell_prior_sig)
    th_ps = _gather(theta_prior_sig)
    n_ps = _gather(sersic_n_prior_sig)

    # A finite Sersic-n prior on a SERSIC row THAWS the free-n gate (in
    # addition to the S1 S/N gate). Threaded through fn_flag so the
    # solver (keep[:,:,5]) AND the host unpack (which gates the fitted n
    # + its error on fn_flag) agree.
    is_ser_m = st == SOURCE_SERSIC
    has_n_prior = is_ser_m & np.isfinite(n_ps) & (n_ps > 0) & mmask
    fn_flag = (free_n_all[midx] | has_n_prior) & mmask

    keep = np.zeros((n_isl, N, 6), dtype=bool)
    keep[:, :, 0:2] = mmask[:, :, None]
    keep[:, :, 2:5] = (is_ext & mmask)[:, :, None]
    keep[:, :, 5] = fn_flag

    p0 = np.stack([sx, sy, lr, e1, e2, u_seed], axis=2
                  ).astype(np.float32)              # (n_isl, N, 6)
    big = np.full((n_isl, N), 1.0, np.float32)
    lower = np.stack([
        sx - pos_max_shift_pix, sy - pos_max_shift_pix,
        np.full((n_isl, N), np.log(0.1), np.float32),
        -100 * big, -100 * big, -12 * big], axis=2).astype(np.float32)
    upper = np.stack([
        sx + pos_max_shift_pix, sy + pos_max_shift_pix,
        np.full((n_isl, N), np.log(0.75 * T), np.float32),
        100 * big, 100 * big, 12 * big], axis=2).astype(np.float32)

    # ---- prior MEAN = seed (p0); prior INV_SIGMA per param ----
    inv = np.zeros((n_isl, N, 6), dtype=np.float64)
    pos_ok = np.isfinite(pe_m) & (pe_m > 0) & mmask
    pos_inv = np.where(pos_ok, 1.0 / np.where(pos_ok, pe_m, 1.0), 0.0)
    inv[:, :, 0] = pos_inv
    inv[:, :, 1] = pos_inv

    ext_ok = is_ext & mmask
    re_m = np.exp(lr.astype(np.float64))
    sig_lr, sig_e1, sig_e2 = _propagate_shape_errors_np(
        e1, e2, re_m, ell_ps, th_ps, re_ps)
    lr_ok = ext_ok & np.isfinite(sig_lr) & (sig_lr > 0)
    inv[:, :, 2] = np.where(lr_ok, 1.0 / np.where(lr_ok, sig_lr, 1.0), 0.0)
    ee_ok = (ext_ok & np.isfinite(sig_e1) & (sig_e1 > 0)
             & np.isfinite(sig_e2) & (sig_e2 > 0))
    inv[:, :, 3] = np.where(ee_ok, 1.0 / np.where(ee_ok, sig_e1, 1.0), 0.0)
    inv[:, :, 4] = np.where(ee_ok, 1.0 / np.where(ee_ok, sig_e2, 1.0), 0.0)

    # Sersic n on the internal u_n: inv_sigma_u = (dn/du) / SERSIC_N_PRIOR_SIG
    # at the SEED n (logit-bijection Jacobian; inverse of #28a's dn/du).
    sn64 = sn.astype(np.float64)
    dndu = ((sn64 - SERSIC_N_MIN) * (SERSIC_N_MAX - sn64)
            / (SERSIC_N_MAX - SERSIC_N_MIN))
    inv[:, :, 5] = np.where(
        has_n_prior, dndu / np.where(has_n_prior, n_ps, 1.0), 0.0)

    prior_mean = p0.copy()
    prior_inv_sigma = inv.astype(np.float32)

    has_priors = bool(np.any(prior_inv_sigma != 0))

    # #35 scene-subtract: per-island member S1 fluxes (n_isl, N, n_bands) for
    # the in-kernel member add-back (run_class_chunk adds Sum_k member_s1_flux
    # * models_band(p0) back to the gathered cutout). s1_flux[midx] gathered +
    # zeroed on padded slots. None (scene_subtract off) -> the kernel guards on
    # its static scene_subtract flag, never on key presence.
    member_s1_flux = None
    if s1_flux is not None:
        # STAR-only add-back: zero galaxy members so the add-back matches the
        # STAR-only field-wide subtract (island_stage subtracts only stars).
        # Extended sources are untouched by scene-subtraction -> they keep
        # their true light and stay in the joint fit (their seed shape is not
        # trusted enough to subtract; deferred to tier-3 iteration). is_ext =
        # st != SOURCE_STAR (computed above).
        member_s1_flux = (
            np.asarray(s1_flux, dtype=np.float32)[midx]
            * (mmask & ~is_ext)[:, :, None]).astype(np.float32)

    out = dict(
        ox=ox, oy=oy, x0b=x0b, y0b=y0b,
        st=st, lr=lr, e1=e1, e2=e2, sn=sn,
        fn_flag=fn_flag, is_ext=is_ext, mmask=mmask, keep=keep,
        gsel=gsel, gvalid=gvalid,
        p0=p0, lower=lower, upper=upper,
        prior_mean=prior_mean, prior_inv_sigma=prior_inv_sigma,
        has_priors=has_priors, member_s1_flux=member_s1_flux,
    )
    if cb_scale is not None:
        # F3-S2: per-island crossband amplitude scale = member-mean of the
        # per-source scale (present ONLY in crossband mode, so band-mode
        # host dicts keep their exact legacy key set).
        _cs = np.asarray(cb_scale, np.float64)[midx]
        out["cb_scale"] = (
            np.where(mmask, _cs, 0.0).sum(axis=1)
            / np.maximum(mmask.sum(axis=1), 1)).astype(np.float32)
    return out


def unpack_class_results(
    plan: IslandClassPlan,
    host: dict,
    island_set,
    res_p: np.ndarray,
    res_f: np.ndarray,
    res_fe: np.ndarray,
    res_chi2: np.ndarray,
    res_conv: np.ndarray,
    res_stall: np.ndarray,
    res_nst: np.ndarray,
    res_perr: np.ndarray | None = None,
    res_pdchi2: np.ndarray | None = None,
    res_reprof: np.ndarray | None = None,
    reprof_deltas: np.ndarray | None = None,
    res_smetrics: np.ndarray | None = None,
    res_swin: np.ndarray | None = None,
    res_femarg: np.ndarray | None = None,
    res_div: np.ndarray | None = None,
    res_railed: np.ndarray | None = None,
    res_ngood: np.ndarray | None = None,
    n_bands: int = 0,
) -> dict:
    """Island-row result arrays -> source-row dict (valid members only).

    ``res_swin`` (#35, optional): (n_isl, n_bands, N) per-band
    data-constrained window fluxes from the source_metrics scan; None
    (metrics off) emits NaN. ``res_div`` (#35, optional): (n_isl,) sticky
    per-island diverged flags; None emits all-False.

    Mirrors the historical solver unpack; used by both the
    single-process solver and the sharded parent.

    ``res_perr`` (task #28a, optional): (n_isl, N, 7) per-member
    ``[var_x, var_y, var_logre, var_ee1, var_ee2, var_un,
    cov_ee1_ee2]`` from the solver's post-convergence marginal
    covariance (``final_errors``: diagonal + the ee1-ee2 off-diagonal
    of ``inv(H)``, flux-marginalized, background frozen). Host-side
    this function (a) masks non-kept slots to NaN (their unit diagonal
    in H reads variance exactly 1.0), (b) maps degenerate-Hessian
    garbage (non-finite or <= 0 variances) to NaN, and (c) propagates
    to the CATALOG frame with exact Jacobians of
    ``ellipse_soft_to_re_ab_phi_np`` (source_model.py):

    - ``Re = exp(logre)``            -> ``sigma_Re = Re * sigma_logre``
    - ``ELL = 1 - ab = 2e/(1+e)``, ``e = 1 - exp(-ee)``,
      ``ee = hypot(ee1, ee2)``:
      ``dELL/d(ee1,ee2) = 2 exp(-ee)/(2 - exp(-ee))^2 * (ee1, ee2)/ee``
    - ``THETA = -deg(atan2(ee2, ee1))/2`` (the mod-180 wrap and the
      -90 deg shift have zero derivative):
      ``dTHETA/d(ee1,ee2) = (90/pi) * (ee2, -ee1)/ee^2``  [deg]
      with ``var_Q = g^T Sigma2 g``,
      ``Sigma2 = [[var_ee1, cov], [cov, var_ee2]]``
    - ``sigma_n = sigma_u * dn/du``,
      ``dn/du = (n - N_MIN)(N_MAX - n)/(N_MAX - N_MIN)`` (the logit
      bijection Jacobian, identical to the legacy joint_varpro
      ``sersic_n_err``); free-n rows only, frozen rows NaN.

    At exactly ``ee == 0`` the (ee1, ee2) direction is undefined (0/0):
    ELL_err and THETA_err are both NaN there. At small ``ee`` THETA_err
    legitimately blows up (~1/ee) and is kept -- that IS the honest
    statistical error of a nearly-round source's position angle.
    ``res_perr=None`` emits NaN for all six error keys.

    ``res_perr`` also carries (in columns 7:13) the 6 per-internal-param
    prior fractions, mapped here to the 6 catalog-frame prior_frac
    disclosure columns (``pos_x/pos_y/re/ell/theta/sersic_n_prior_frac``):
    0 = pure data, 1 = pure prior, NaN where that param carries no prior /
    is not fitted. ``Re``/``sersic_n`` prior_frac equal the
    ``logre``/``u_n`` internal fractions (a monotone reparam does not
    change the prior/data information ratio). ``ELL``/``THETA`` prior_frac
    are the variance-contribution-weighted averages of the ee1/ee2
    internal fractions with the SAME weights each output's error
    propagation uses (ELL radial ``(ee1^2, ee2^2)/ee^2``; THETA tangential
    ``(ee2^2, ee1^2)/ee^2``) -- a documented, constant-free choice that
    reduces to the scalar formula when the two ee fractions are equal and
    stays in [0, 1]. ``res_perr=None`` emits NaN. (Per-band FLUX priors,
    and their prior_frac, are DEFERRED to the P7 field-level flux solve.)
    """
    midx = plan.midx
    mmask = host["mmask"]
    is_ext = host["is_ext"]
    lr, e1, e2, sn = host["lr"], host["e1"], host["e2"], host["sn"]
    fn_flag = host["fn_flag"]
    isl_ids = plan.isl_ids

    ii, kk = np.nonzero(mmask)
    # sersic_u_to_n twin (sigmoid back onto the open range)
    u = res_p[:, :, 5]
    n_free = SERSIC_N_MIN + (SERSIC_N_MAX - SERSIC_N_MIN) / (1.0 + np.exp(-u))
    n_final = np.where(fn_flag, n_free, sn).astype(np.float32)

    # ---- #28a: variances -> catalog-frame 1-sigma errors ----
    n_mem = ii.size
    if res_perr is None:
        pos_x_err = np.full(n_mem, np.nan, np.float32)
        pos_y_err = pos_x_err.copy()
        re_err = pos_x_err.copy()
        ell_err = pos_x_err.copy()
        theta_err = pos_x_err.copy()
        nerr = pos_x_err.copy()
    else:
        keep_m = host["keep"][ii, kk, :]                    # (n_mem, 6)
        var = res_perr[ii, kk, :6].astype(np.float64)       # (n_mem, 6)
        cov = res_perr[ii, kk, 6].astype(np.float64)        # (n_mem,)
        ok = keep_m & np.isfinite(var) & (var > 0)

        def _sig(j):
            return np.where(ok[:, j],
                            np.sqrt(np.where(ok[:, j], var[:, j], 1.0)),
                            np.nan)

        pos_x_err = _sig(0).astype(np.float32)
        pos_y_err = _sig(1).astype(np.float32)
        # Re = exp(logre): exact Jacobian (no q dependence in
        # ellipse_soft_to_re_ab_phi_np).
        re_fit = np.exp(res_p[ii, kk, 2].astype(np.float64))
        re_err = (re_fit * _sig(2)).astype(np.float32)
        # ELL / THETA: exact 2x2 quadratic forms with the ee1-ee2
        # covariance (formulas in the docstring).
        e1v = res_p[ii, kk, 3].astype(np.float64)
        e2v = res_p[ii, kk, 4].astype(np.float64)
        ee = np.hypot(e1v, e2v)
        ok2 = ok[:, 3] & ok[:, 4] & np.isfinite(cov) & (ee > 0)
        ee_s = np.where(ee > 0, ee, 1.0)                    # 0/0 guard
        q_ell = (e1v * e1v * var[:, 3] + 2.0 * e1v * e2v * cov
                 + e2v * e2v * var[:, 4]) / (ee_s * ee_s)
        q_th = (e2v * e2v * var[:, 3] - 2.0 * e1v * e2v * cov
                + e1v * e1v * var[:, 4]) / (ee_s ** 4)
        exp_nee = np.exp(-ee)
        dell_dee = 2.0 * exp_nee / (2.0 - exp_nee) ** 2
        var_ell = dell_dee * dell_dee * q_ell
        var_th = (90.0 / np.pi) ** 2 * q_th

        def _sigq(v, okq):
            good = okq & np.isfinite(v) & (v > 0)
            return np.where(good, np.sqrt(np.where(good, v, 1.0)),
                            np.nan).astype(np.float32)

        ell_err = _sigq(var_ell, ok2)
        theta_err = _sigq(var_th, ok2)
        # Sersic n: logit-bijection chain rule at the FITTED n
        # (legacy solver_grid._varpro_sersic_n_err convention).
        n_m = n_final[ii, kk].astype(np.float64)
        dndu = ((n_m - SERSIC_N_MIN) * (SERSIC_N_MAX - n_m)
                / (SERSIC_N_MAX - SERSIC_N_MIN))
        nerr = np.where(fn_flag[ii, kk],
                        _sig(5) * np.abs(dndu), np.nan).astype(np.float32)

    # ---- R1 prior_frac disclosure (per catalog-frame nonlinear param) ----
    pinv = host["prior_inv_sigma"]                      # (n_isl, N, 6)
    if res_perr is None:
        pf_x = np.full(n_mem, np.nan, np.float32)
        pf_y = pf_x.copy(); re_pf = pf_x.copy()
        ell_pf = pf_x.copy(); theta_pf = pf_x.copy(); n_pf = pf_x.copy()
    else:
        pf6 = res_perr[ii, kk, 7:13].astype(np.float64)   # (n_mem, 6)
        pinv_m = pinv[ii, kk, :]                            # (n_mem, 6)
        keep_m = host["keep"][ii, kk, :]                   # (n_mem, 6)
        has_p = pinv_m > 0                                 # per internal param

        def _pf(j):
            good = keep_m[:, j] & has_p[:, j]
            return np.where(good, pf6[:, j], np.nan).astype(np.float32)

        pf_x = _pf(0)
        pf_y = _pf(1)
        re_pf = _pf(2)          # Re prior_frac == logre's (monotone reparam)
        n_pf = np.where(fn_flag[ii, kk] & has_p[:, 5],
                        pf6[:, 5], np.nan).astype(np.float32)
        # ELL / THETA: variance-contribution-weighted average of the
        # ee1/ee2 internal prior fractions (radial vs tangential weights).
        e1v = res_p[ii, kk, 3].astype(np.float64)
        e2v = res_p[ii, kk, 4].astype(np.float64)
        ee2v = e1v * e1v + e2v * e2v
        shp = (keep_m[:, 3] & keep_m[:, 4] & has_p[:, 3] & has_p[:, 4]
               & (ee2v > 0))
        ee2s = np.where(ee2v > 0, ee2v, 1.0)
        ell_pf = np.where(
            shp, (e1v * e1v * pf6[:, 3] + e2v * e2v * pf6[:, 4]) / ee2s,
            np.nan).astype(np.float32)
        theta_pf = np.where(
            shp, (e2v * e2v * pf6[:, 3] + e1v * e1v * pf6[:, 4]) / ee2s,
            np.nan).astype(np.float32)

    # ---- LEG D nonlinear (task #20): per-island reduced chi2 ----
    # chi2_nu = chi2 / dof, dof = n_good_pixels - n_free_params, with
    # n_free = (kept nonlinear params: keep & mmask) + n_bands * (active
    # members). PER-ISLAND (shared by members): the residual/chi2 is joint
    # over the island, so a shared reduced chi2 is the honest quantity for a
    # blend. Uses the SAME sky-marginal whitened residual as the fit, so it
    # reads ~1 under pure correlated noise (no double-count with LEG C) and
    # inflates on confusion / model mismatch. When priors are active the
    # island chi2 includes the (small) nonlinear prior penalty -- a
    # conservative addition documented in IslandFitResult.chi2_nu.
    if res_ngood is not None and int(n_bands) > 0:
        n_isl_c = int(mmask.shape[0])
        keep_eff = host["keep"] & mmask[:, :, None]          # (n_isl, N, 6)
        keep_isl = keep_eff.reshape(n_isl_c, -1).sum(axis=1)
        mm_isl = mmask.sum(axis=1)                            # active members
        n_free_isl = keep_isl.astype(np.float64) + \
            float(n_bands) * mm_isl.astype(np.float64)
        dof_isl = res_ngood.astype(np.float64) - n_free_isl
        with np.errstate(invalid="ignore", divide="ignore"):
            chi2_nu_isl = np.where(
                (dof_isl > 0) & np.isfinite(res_chi2),
                res_chi2 / np.maximum(dof_isl, 1.0), np.nan)
        chi2_nu = chi2_nu_isl[ii].astype(np.float32)
    else:
        chi2_nu = np.full(ii.size, np.nan, np.float32)

    # ---- #19a: Re profile-likelihood Delta-chi2<=1 interval ----
    if res_reprof is None or reprof_deltas is None:
        re_prof_lo = np.full(n_mem, np.nan, np.float32)
        re_prof_hi = re_prof_lo.copy()
    else:
        _dg = np.asarray(reprof_deltas, np.float64)
        _ord = np.argsort(_dg)
        _dgs = _dg[_ord]
        _prof = res_reprof[ii, kk, :].astype(np.float64)[:, _ord]
        _prof = np.where(np.isfinite(_prof), _prof, np.inf)

        def _cross(pn, dn):
            # pn: (n_mem, Ks) profile walked OUTWARD from delta=0; dn: (Ks,)
            # signed offsets in the same order. Returns the interpolated
            # crossing of Delta-chi2 = 1 (clamped at the outer edge).
            P = np.concatenate(
                [np.zeros((pn.shape[0], 1)), pn], axis=1)
            D = np.concatenate([[0.0], dn])
            above = P >= 1.0
            has = above.any(axis=1)
            j = np.clip(np.argmax(above, axis=1), 1, None)
            p0 = np.take_along_axis(P, (j - 1)[:, None], 1)[:, 0]
            p1 = np.take_along_axis(P, j[:, None], 1)[:, 0]
            d0 = D[j - 1]
            d1 = D[j]
            t = np.where(p1 > p0, (1.0 - p0)
                         / np.maximum(p1 - p0, 1e-30), 1.0)
            return np.where(has, d0 + np.clip(t, 0.0, 1.0) * (d1 - d0),
                            dn[-1])

        _neg = _dgs < 0
        _pos = _dgs > 0
        # negative side walked outward = reversed ascending order
        _dlo = _cross(_prof[:, _neg][:, ::-1], _dgs[_neg][::-1])
        _dhi = _cross(_prof[:, _pos], _dgs[_pos])
        _re_m = np.exp(res_p[ii, kk, 2].astype(np.float64))
        _extm = is_ext[ii, kk]
        re_prof_lo = np.where(_extm, _re_m * np.exp(_dlo),
                              np.nan).astype(np.float32)
        re_prof_hi = np.where(_extm, _re_m * np.exp(_dhi),
                              np.nan).astype(np.float32)

    return dict(
        src=midx[ii, kk],
        pos_x=res_p[ii, kk, 0], pos_y=res_p[ii, kk, 1],
        logre=np.where(is_ext[ii, kk], res_p[ii, kk, 2], lr[ii, kk]),
        ee1=np.where(is_ext[ii, kk], res_p[ii, kk, 3], e1[ii, kk]),
        ee2=np.where(is_ext[ii, kk], res_p[ii, kk, 4], e2[ii, kk]),
        sersic_n=n_final[ii, kk],
        fluxes=res_f[ii, :, kk],
        flux_err=res_fe[ii, :, kk],
        chi2=res_chi2[ii], chi2_nu=chi2_nu, n_steps=res_nst[ii],
        converged=res_conv[ii],
        stalled=res_stall[ii],
        diverged=(res_div[ii] if res_div is not None
                  else np.zeros(ii.size, dtype=bool)),
        stall_dchi2=(res_pdchi2[ii].astype(np.float32)
                     if res_pdchi2 is not None
                     else np.full(ii.size, np.nan, np.float32)),
        railed=(res_railed[ii] if res_railed is not None
                else np.zeros(ii.size, dtype=bool)),
        free_n=fn_flag[ii, kk],
        island_id=isl_ids[ii].astype(np.int32),
        island_nsrc=island_set.n_src[isl_ids][ii].astype(np.int32),
        island_giant=np.zeros(ii.size, dtype=bool),
        # Per-member fitted-Re upper bound in pixels (0.75 * stamp TIER;
        # host["upper"][...,2] = log(0.75*T)). The tier is chosen from the
        # seed shape, so a promoted-tier (111/221) island has a LARGER Re
        # ceiling than the native 0.75*55 -- the pipeline uses this to make
        # flag_shape_bound tier-aware instead of comparing to the scalar
        # native ceiling.
        re_upper_pix=np.exp(
            host["upper"][ii, kk, 2].astype(np.float64)).astype(np.float32),
        pos_x_err=pos_x_err, pos_y_err=pos_y_err,
        re_err=re_err, ell_err=ell_err, theta_err=theta_err,
        re_prof_lo=re_prof_lo, re_prof_hi=re_prof_hi,
        sersic_n_err=nerr,
        pos_x_prior_frac=pf_x, pos_y_prior_frac=pf_y,
        re_prior_frac=re_pf, ell_prior_frac=ell_pf,
        theta_prior_frac=theta_pf, sersic_n_prior_frac=n_pf,
        # #17 per-source metrics: (n_isl, N, 4) = [rchisq, fracflux, fracin,
        # fracmasked]; per-member via the same [ii, kk] valid-slot indexing.
        # res_smetrics=None (knob off) -> NaN for all four.
        rchisq=(res_smetrics[ii, kk, 0] if res_smetrics is not None
                else np.full(ii.size, np.nan, np.float32)),
        fracflux=(res_smetrics[ii, kk, 1] if res_smetrics is not None
                  else np.full(ii.size, np.nan, np.float32)),
        fracin=(res_smetrics[ii, kk, 2] if res_smetrics is not None
                else np.full(ii.size, np.nan, np.float32)),
        fracmasked=(res_smetrics[ii, kk, 3] if res_smetrics is not None
                    else np.full(ii.size, np.nan, np.float32)),
        # #35: per-band window fluxes -- same [ii, :, kk] advanced-index
        # convention as `fluxes` -> (n_mem, n_bands). None (metrics off)
        # -> NaN.
        flux_win=(res_swin[ii, :, kk] if res_swin is not None
                  else np.full((ii.size, max(int(n_bands), 0)), np.nan,
                               np.float32)),
        # D2: per-band SHAPE-marginalized flux-error INCREMENT variance --
        # same [ii, :, kk] convention -> (n_mem, n_bands). 0 for star rows
        # (no free shape block); None (marginal-flux off) -> NaN.
        flux_marg_var=(res_femarg[ii, :, kk] if res_femarg is not None
                       else np.full((ii.size, max(int(n_bands), 0)), np.nan,
                                    np.float32)),
    )


def giant_seed_rows(giant_ids, island_set, *, pos_x, pos_y, logre, ee1, ee2,
                    sersic_n, n_bands: int) -> list:
    """Per-source rows for giant islands: geometry kept at the catalog seed,
    nothing solved (plan R16, 2026-09-26).

    A giant island (larger than the biggest bucket class, more members than
    ``joint_nsrc_cap``, or measured memory-infeasible one island at a time)
    gets no joint nonlinear fit. Its members keep the seed position and
    shape, and pass 2 measures their fluxes at that geometry exactly as for
    every other source, so the island stage has nothing to compute for them.
    ``fluxes``/``flux_err`` are NaN: the island stage does not measure them.

    The removed host flux-only solve (``_fit_giants``) produced numbers that
    nothing read, that were wrong under ``scene_subtract`` (no member
    add-back), that cost ~480 s of worker time per crowded-anchor run, and
    whose dense per-member canvas asked for 854 GiB on T09469.

    One dict per island, sorted by island id (run-independent row order),
    with the same keys as :func:`unpack_class_results`."""
    rows = []
    for g in sorted(int(i) for i in np.asarray(giant_ids).ravel()):
        src = np.asarray(island_set.members[g], dtype=np.int64)
        n = int(src.size)

        def nan_n():
            return np.full(n, np.nan, np.float32)

        def nan_nb():
            return np.full((n, max(int(n_bands), 0)), np.nan, np.float32)

        rows.append(dict(
            src=src,
            pos_x=np.asarray(pos_x)[src].astype(np.float32),
            pos_y=np.asarray(pos_y)[src].astype(np.float32),
            logre=np.asarray(logre)[src].astype(np.float32),
            ee1=np.asarray(ee1)[src].astype(np.float32),
            ee2=np.asarray(ee2)[src].astype(np.float32),
            sersic_n=np.asarray(sersic_n)[src].astype(np.float32),
            fluxes=nan_nb(), flux_err=nan_nb(),
            # No island fit: no chi2, and chi2_nu is undefined (the pipeline
            # floors sqrt(max(1, NaN)) to no inflation).
            chi2=nan_n(), chi2_nu=nan_n(),
            n_steps=np.zeros(n, np.int32),
            converged=np.ones(n, bool),
            stalled=np.zeros(n, bool),
            diverged=np.zeros(n, bool),
            stall_dchi2=nan_n(),
            railed=np.zeros(n, bool),
            free_n=np.zeros(n, bool),
            island_id=np.full(n, g, np.int32),
            island_nsrc=np.full(n, n, np.int32),
            island_giant=np.ones(n, bool),
            # No fitted Re bound -> NaN; the pipeline falls back to the
            # native ceiling.
            re_upper_pix=nan_n(),
            # No nonlinear Hessian, no priors, no model scan, no window
            # flux, no shape marginalization: all undefined (NaN).
            pos_x_err=nan_n(), pos_y_err=nan_n(),
            re_err=nan_n(), ell_err=nan_n(),
            re_prof_lo=nan_n(), re_prof_hi=nan_n(),
            theta_err=nan_n(), sersic_n_err=nan_n(),
            pos_x_prior_frac=nan_n(), pos_y_prior_frac=nan_n(),
            re_prior_frac=nan_n(), ell_prior_frac=nan_n(),
            theta_prior_frac=nan_n(), sersic_n_prior_frac=nan_n(),
            rchisq=nan_n(), fracflux=nan_n(),
            fracin=nan_n(), fracmasked=nan_n(),
            flux_win=nan_nb(),
            flux_marg_var=nan_nb(),
        ))
    return rows


__all__ = [
    "DEFAULT_MAX_STEPS_MAIN",
    "ladder_width",
    "HARD_CHUNK_DIV",
    "IslandFitResult",
    "IslandClassPlan",
    "plan_island_classes",
    "class_host_arrays",
    "unpack_class_results",
    "giant_seed_rows",
    "free_n_gate",
    "sersic_n_to_u_np",
    "_CHUNK_OVERRIDE",
    "_TC_OVERRIDE",
]
