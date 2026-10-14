"""Per-island VarPro-LM solver (island-architecture P2+P3).

Fits every bucketed island of a field -- singletons AND multi-source
islands -- with a per-island damped VarPro LM step, vmapped over the
islands of one (bbox_class, nsrc_class, star|mixed) bucket. Giant
islands (exceeding the largest class) are not fitted: their members keep
the seed geometry (:func:`.island_plan.giant_seed_rows`; plan R16) and
pass 2 measures their fluxes. Design:
the internal design note ``DESIGN_island_arch.md`` (not distributed) stages IF/BK, phases
P2 (singletons) + P3 (multi-source).

Inputs are the state after the existing S1 global solve: per-band
``data - bg_global`` images (bg FROZEN per design D1), invvar (masked
pixels -> 0), the per-band polynomial PSFs, catalog seeds, and the
:mod:`.islands` decomposition built from S1 fluxes (kappa=0.5).

Model/solver conventions deliberately mirror production:

- stamps rendered by :func:`.source_model.render_source_stamp_at`
  (same Lanczos + round placement, same fft/direct backends, NO star
  renorm -- matching the mixed-source S1 path), at a per-bucket tier
  ``T = next odd >= max(bbox_class, native)`` via the padded-PSF trick;
  members are placed onto the island bbox with the production
  ``stamp_top_left`` convention (gather-sum, not scatter -- equivalent
  and vmap/deterministic-friendly);
- per-band member fluxes are CLOSED FORM inside the residual (VarPro):
  the (nsrc x nsrc) weighted normal matrix over the island bbox is
  solved in FLOAT64 (like production's linear solves); padded member
  slots contribute zero rows/cols with a unit diagonal so the solve
  stays PD; AD flows through ``jnp.linalg.solve``;
- nonlinear vector = 6 x nsrc with per-member keep-masks (STAR: x,y;
  extended: +logre,ee1,ee2; TYPE=SERSIC passing the S1 S/N gate
  ``n_fit_min_snr``: +u_n) -- the per-island dense Hessian is
  (6*nsrc)^2 <= 384^2, solved in float64;
- bounds and per-step clips match the production varpro step;
- per-island convergence (P3b policy, three prongs -- an island is
  converged when ANY fires):
    1. RELATIVE dchi2: ``dchi2 < ftol_rel*max(chi2,1)`` on 2
       consecutive ACCEPTED steps. The P2 absolute ftol (0.1) demanded
       dchi2/chi2 ~ 5e-6 at 20k-pixel islands -- weakly-constrained
       free-n params wander at the noise floor with dchi2 O(0.1-1)
       indefinitely (measured: 100% of T08285's 90 free-n singletons
       hit max_steps at S/N~50). Relative 1e-4 (dchi2 ~ 2 at 20k px,
       ~1-sigma-scale single-parameter moves) is the plateau detector.
    2. PER-PARAM step tolerances on an accepted step: every member's
       position step < 0.1 px AND every |dlogre|/|dee1|/|dee2|/|du_n|
       < 0.01 (astrometry stays gated by the 0.1 px criterion; shapes
       get their own, tighter scale).
    3. STALL: ``STALL_REJECT_K`` (4) consecutive REJECTED steps ->
       converged with the per-island ``stalled`` flag set (LM at a
       local plateau where every damped step is uphill: measured on
       T08285 mixed islands, chi2 stops improving after step ~5 but
       accept/reject ping-pong burned all 30 steps before P3b).
       #35 stall anatomy (2026-07-11, rootfix_a2 trajectory forensics):
       after a long accept run lam has bottomed at ~1e-9 and the raw
       Newton step is far beyond the per-step clips, so the x10 lam
       escalation cannot change the clip/bound-limited proposal -- the
       4 rejects then test ONE frozen proposal, not four. The solver
       therefore reports proposal identity across the stall's rejects
       (exact comparison, constant-free) as the per-island ``diverged``
       disclosure flag. Default mode: reconstructed POST-HOC by
       ``diverged_probe`` from the stalled FINAL state (rejects never
       move p and lam is x10 per reject, so the K tested proposals are
       a pure function of the final state) -- measured: ANY change to
       the step program itself, even aux-only outputs of existing
       tensors, shifts XLA fusion/autotuning at the ulp level and
       flips stall-marginal trajectories, so the default step program
       stays byte-identical to the pre-#35 solver. Opt-in "changed"
       mode, whose counting rule needs the signal in-flight, carries
       the tracker in-kernel. ``stall_count_mode``
       ("proposals" = legacy default: every reject counts; "changed" =
       opt-in semantic fix: a reject counts only when the clipped
       proposal DIFFERS from the previously rejected one, so identical
       rejects keep escalating lam without counting). CAUTION: in
       crowded fields the chi2 minimum itself can be DISPLACED by
       uncataloged ambient light (model incompleteness) -- "changed"
       rides such islands deeper into the wrong minimum; pair it with
       the diverged flag + FLUX_WIN disclosure (docs/outputs.md).
  Islands unconverged after ``max_steps`` land on the re-queue list
  (survey note S5).

ENGINEERING RULES (P2 lessons -- keep these invariants):

1. EVERY jitted kernel is cached module-wide on its SHAPE key, with all
   per-field data (incl. PSF polynomials) flowing in as traced
   arguments -- never close over device arrays and never call
   ``jax.jit`` on a per-call closure (a fresh jit per gather call cost
   ~15 s/compile and dominated the entire P2 wall time).
2. Warm ``sersic_mog.get_table_jax()`` EAGERLY before any fft-render
   trace: its lazy module cache would capture a tracer (see TODO in
   sersic_mog.py).
3. Chunk sizes come from an explicit peak-memory model (render-tangent
   intermediates + Jacobian) -- Jacobian tangents are pushed in chunks
   for large nsrc so intermediates stay bounded.
4. Prefer ISLAND parallelism over TANGENT parallelism (P3b): the
   ``lax.map`` over tangent segments is batched over the whole island
   chunk, so a small ``tangent_chunk`` (6 = one member) costs nothing
   in occupancy while dividing the per-island working set -- the old
   largest-fitting-tangent-chunk rule (24) inflated per-island memory
   ~4x and forced chunk(s)-of-1 exactly on the nsrc>=16 buckets where
   vmap parallelism matters most.

Everything is jit+vmap (no scatter-adds), so results are deterministic
bit-for-bit across passes WITHIN a process. They are NOT invariant to
the island chunk width (P5 measurement, 2026-07-06): XLA compiles a
different program per vmap batch size and the per-lane reduction
order shifts at the ulp level, which the LM accept/reject boundary
occasionally amplifies into a trajectory flip on stall-marginal
blends -- the same magnitude class as the accepted production
run-to-run noise. Run with JAX_ENABLE_X64=1 (production worker
default) so the float64 inner solves are real float64.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .source_model import (
    SOURCE_SERSIC, SOURCE_STAR,
    render_source_stamp_at, sersic_n_to_u, sersic_u_to_n,
)


# LM / convergence defaults (mirroring production knobs where they
# exist). DEFAULT_MAX_STEPS / DEFAULT_CHI2_FTOL_REL / DEFAULT_STEP_TOL_PIX
# / DEFAULT_POS_STEP_MAX / DEFAULT_UN_FREEZE_STEPS live in island_plan.py
# (#25) so the JAX-free stage parent shares them; imported below.
# DEFAULT_UN_FREEZE_STEPS rationale (P3b): u_n annealing budget (freeze
# u_n after k steps; 0 = never). Default OFF: with the round-galaxy
# Jacobian singularity fixed (see models_band()), the A/B experiment
# (analysis/islandP3b_convexp2.log) shows free-n islands converge in
# 2-7 steps and annealing at k=8 never binds (B-vs-A |dn| max 0.001,
# chi2 identical to 7e-8) -- kept as an opt-in safety lever for
# genuinely degenerate fields.
from .island_plan import (  # noqa: F401  (shared defaults, re-exported)
    DEFAULT_CHI2_FTOL_REL, DEFAULT_MAX_STEPS, DEFAULT_POS_STEP_MAX,
    DEFAULT_STEP_TOL_PIX, DEFAULT_UN_FREEZE_STEPS,
    REPROF_DELTAS_LN,
)

DEFAULT_SHAPE_STEP_TOL = 0.01    # P3b: per-param dlogre/dee1/dee2/du_n tol
STALL_REJECT_K = 4               # P3b: k consecutive rejects -> stalled
DEFAULT_SHAPE_STEP_MAX = 0.15    # production varpro max_shape_step
DEFAULT_UN_STEP_MAX = 0.5
LM_LAMBDA0 = 1e-3                # production damp seed
LM_UP = 10.0                     # production per-source damping adapts 10x
LM_DOWN = 0.1

# Trust-region (step_mode="trust_region", opt-in) radius-update constants.
# These are the ALGORITHM-STANDARD Nocedal & Wright "Numerical Optimization"
# 2e Alg. 4.1 / Moré (1978) MINPACK trust-region thresholds -- dataset-
# INDEPENDENT (unlike a tuned photometric constant): shrink the radius when
# the actual/predicted chi2 gain ratio rho < TR_ETA_LO, expand (if the step
# was at the region boundary) when rho > TR_ETA_HI.
TR_ETA_LO = 0.25
TR_ETA_HI = 0.75
TR_SHRINK = 0.25
TR_EXPAND = 2.0

# LM state carried from a chunk's main pass into its re-queued hard
# chunk (exact-continuation semantics; see run_class_chunk). #35 adds
# diverged / n_same / prej so the CHANGED-mode in-kernel tracker state
# (sticky diverged flag, identical-proposal run length, last rejected
# proposal) survives the main->hard re-queue exactly; in the default
# mode these carry zeros (diverged is reconstructed post-hoc per chunk
# and a stalled island is conv -> never re-queued). KEEP IN SYNC with
# the literal tuple in island_stage._run_island_fit_sharded (JAX-free
# parent).
_CARRY_KEYS = ("p", "lam", "chi2", "consec", "nrej", "nst",
               "diverged", "n_same", "prej")

# Peak-memory chunk model + class planning: moved to island_plan.py
# (island-arch #25) so the JAX-free pipeline parent can plan/price the
# multi-GPU chunk queue. Re-exported here for the analysis probes
# (analysis/island_p3b_memprobe.py imports them from this module).
from .island_plan import (  # noqa: F401  (re-exports)
    DEFAULT_MAX_STEPS_MAIN, HARD_CHUNK_DIV, _CHUNK_OVERRIDE, _TC_OVERRIDE,
    class_host_arrays, free_n_gate, giant_seed_rows, ladder_width,
    plan_island_classes, unpack_class_results,
)


def _next_odd(v: int) -> int:
    v = int(v)
    return v if v % 2 == 1 else v + 1


def _pad_psf_np(psf_obj, tier: int):
    """Zero-pad a numpy-side PsfPolynomial to (tier, tier) -- the
    :func:`.runner._maybe_pad_psf` pattern (odd tiers keep the center
    pixel convention ``H//2`` exact)."""
    import dataclasses
    H, W = int(psf_obj.H_psf), int(psf_obj.W_psf)
    if tier <= H and tier <= W:
        return psf_obj
    coefs = psf_obj.coefs
    pad_y0 = (tier - H) // 2
    pad_x0 = (tier - W) // 2
    new_coefs = np.zeros((tier, tier) + coefs.shape[2:], dtype=coefs.dtype)
    new_coefs[pad_y0:pad_y0 + H, pad_x0:pad_x0 + W] = coefs
    return dataclasses.replace(psf_obj, coefs=new_coefs, H_psf=tier, W_psf=tier)


# IslandFitResult moved to island_plan.py (island-arch #25) so the
# JAX-free pipeline parent can assemble sharded results; re-exported
# here for back-compat.
from .island_plan import IslandFitResult  # noqa: F401

# Backward-compat alias (P2 name).
SingletonFitResult = IslandFitResult


_KERNEL_CACHE: dict[tuple, tuple] = {}


def _make_bucket_kernels(
    *,
    B: int,
    T: int,
    N: int,
    n_bands: int,
    n_terms: int,
    tangent_chunk: int,
    render_kind: str,          # "star" | "mixed"
    n_gal: int,                # PERF #14: FFT sub-batch width G (mixed)
    star_render_split: bool,   # PERF #14: enable the star/galaxy split
    render_backend: str,
    reg: float,
    pos_step_max: float,
    shape_step_max: float,
    un_step_max: float,
    ftol_rel: float,
    step_tol: float,
    shape_step_tol: float,
    un_freeze_steps: int,
    has_priors: bool = False,
    stall_count_mode: str = "proposals",
    step_mode: str = "clip",
    scene_subtract: bool = False,
    crossband: dict | None = None,
    convergence: str = "legacy",
    plateau_dlnp_tol: float = 0.5,
):
    """Build (gather, step_v, chi2_v, final_v, err_v, residual_v,
    metrics_v, diverged_v, step_protocol, fluxmarg_v, railed_v,
    step_mode, addback_v) for one (bbox_class B,
    nsrc_class N, kind) bucket shape -- the trailing string tells
    run_class_chunk which step protocol the tuple carries.

    ``convergence`` (dlnp campaign 2026-07-30, static -- baked into the
    cache key): "legacy" (default) emits the pre-existing step programs
    VERBATIM (byte-identical jaxpr; the off-discipline of has_priors/
    scene_subtract/crossband). "dlnp" swaps in ``one_step_dlnp``: a
    float64 chi2 CARRY (residual math stays f32 -- E0-validated
    arithmetic; removes the measured fp32 accept coin-flip at
    chi2 >~ 1e6-1e7 / the crossband Pool-2 stall inflation), the
    ABSOLUTE statistical ftol prong (accepted dchi2 < 2*plateau_dlnp_
    tol, the island-size-independent 1-sigma quantum, replacing the
    relative 1e-4 prong that stopped big islands ~20 quanta/step
    mid-descent), a QUANTUM-GATED prong 2 (step tolerances fire only
    when the accepted step is also sub-quantum -- E0: bright-star
    step-exits were the largest leaked-descent channel), the
    distinct-proposal stall counting rule (identical-proposal rejects
    escalate lam x10 without counting -- subsumes stall_count_mode
    "changed", which is therefore an invalid combination), and terminal
    no-information reject-exits (a reject tested AT the lam cap, or a
    NO-MOVE proposal whose damped update rounds to zero against the f32
    params -- either way every future proposal is provably
    bit-identical). Every reject-exit gets the POST-HOC
    ``plateau_probe`` ladder certificate with the SAME f32 arithmetic
    as legacy mode; certificate-PASSING exits are mapped to
    CONVERGED-ON-PLATEAU in run_class_chunk (E1a: under exact f64
    accepts a finished island's natural terminal state IS a reject run
    -- measured 210/220 UDS reject-exits certify at dchi2 ~ 0 with
    legacy-identical endpoints), certificate-FAILING exits stay
    island_stalled (honest stall; hard-fail semantics unchanged).
    Requires JAX x64 (the production worker default) and step_mode
    "clip".

    ``stall_count_mode`` (#35, static -- baked into the cache key):
    "proposals" (legacy default) counts EVERY rejected step toward the
    STALL_REJECT_K stall; ``step_v`` is then BYTE-IDENTICAL to the
    pre-#35 program (measured: any change to the step program, even
    aux-only outputs, shifts XLA fusion/autotuning at the ulp level and
    flips stall-marginal trajectories), and the diverged disclosure is
    reconstructed POST-HOC by ``diverged_v`` on the stalled islands'
    final state. "changed" counts a reject only when the clip/bound-
    limited proposal differs from the previously rejected one
    (identical-proposal rejects keep escalating lam without counting);
    that rule needs the proposal-identity signal inside the program, so
    ``step_v`` is the extended 11-state in-kernel-tracker program
    (opt-in, science-affecting by design).
    ``residual_v`` (task #34) is the per-island invvar-weighted band-coadded
    model-subtracted residual + coadd weight, used ONLY when a re-detect
    pass is requested (``want_residual``); it never runs on the off path.

    See ENGINEERING RULES in the module docstring: cached module-wide on
    the shape key; PSFs flow in as a traced pytree.

    ``step_fn`` performs one joint LM step over the 6*N member params
    with accept/reject + per-island damping + convergence bookkeeping;
    converged islands are frozen (state passes through).

    ``has_priors`` (static) toggles the catalog-driven NONLINEAR Gaussian
    prior injection (position/shape/Sersic-n on the LM step + chi2 penalty
    + posterior covariance). It is baked into the cache key so a class
    with NO priors (``has_priors=False``) emits the pre-prior program
    VERBATIM (byte-identical jaxpr -> bit-identical output); a class WITH
    any prior gets the injected program. The prior arithmetic is purely
    ADDITIVE with an ``inv_sigma == 0`` no-op sentinel, and it lives
    ENTIRELY OUTSIDE the residual/hg_scan (``residual_band`` is now always
    ``(B*B,)`` -- the per-band flux prior that once appended N residual
    rows is DEFERRED to the P7 field-level flux solve). Consequently an
    unpriored member/island within a has_priors=True class is
    bit-identical to the no-prior kernel INCLUDING its chi2 (the ``pen``
    term is exactly zero and adding ``0.0`` / ``diag(0)`` is exact; no
    residual-tree reassociation remains).
    """
    import types

    import jax
    import jax.numpy as jnp

    if stall_count_mode not in ("proposals", "changed"):
        raise ValueError(
            f"stall_count_mode must be 'proposals' or 'changed', "
            f"got {stall_count_mode!r}")
    if step_mode not in ("clip", "trust_region"):
        raise ValueError(
            f"step_mode must be 'clip' or 'trust_region', got {step_mode!r}")
    if convergence not in ("legacy", "dlnp"):
        raise ValueError(
            f"convergence must be 'legacy' or 'dlnp', got {convergence!r}")
    if convergence == "dlnp":
        # Config load already rejects these combinations; re-checked here
        # so direct library callers (island_fit / replay drivers) fail
        # loudly too.
        if step_mode != "clip":
            raise ValueError(
                "convergence='dlnp' requires step_mode='clip' "
                "(trust-region support pending)")
        if stall_count_mode == "changed":
            raise ValueError(
                "convergence='dlnp' subsumes stall_count_mode='changed' "
                "(distinct-proposal counting is the dlnp behavior); "
                "leave stall_count_mode at 'proposals'")
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "convergence='dlnp' needs JAX x64 for the float64 chi2 "
                "carry -- set JAX_ENABLE_X64=1 (the production worker "
                "default)")
    # F3-S1 (crossband GLS): the rank-r ambient term is baked into the
    # traced program as constants -> the cache key carries the VALUES.
    # crossband=None emits the pre-F3 program VERBATIM (byte-identical
    # jaxpr; same off-discipline as has_priors).
    if crossband is not None:
        _cb_key = ("cb", int(crossband["rank"]),
                   np.asarray(crossband["u"], np.float32).tobytes(),
                   np.asarray(crossband["lam"], np.float32).tobytes())
    else:
        _cb_key = ("cb0",)
    # dlnp campaign: the mode (and, in dlnp mode only, the quantum
    # threshold that enters the step program) is static. Legacy runs
    # share one key element regardless of plateau_dlnp_tol -- the
    # tolerance stays host-side there.
    if convergence == "dlnp":
        _conv_key = ("dlnp", float(plateau_dlnp_tol))
    else:
        _conv_key = ("conv0",)
    key = (B, T, N, n_bands, n_terms, int(tangent_chunk),
           render_kind, int(n_gal), bool(star_render_split),
           render_backend, float(reg), float(pos_step_max),
           float(shape_step_max), float(un_step_max), float(ftol_rel),
           float(step_tol), float(shape_step_tol), int(un_freeze_steps),
           bool(has_priors), str(stall_count_mode), str(step_mode),
           bool(scene_subtract), _cb_key, _conv_key)
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    nP = 6 * N
    reg32 = jnp.float32(reg)

    # ---- effarc2 K3 (2026-08-26): STATIC dead-parameter elimination -------
    # A STAR member's shape block is not merely masked, it is CONSTANT in the
    # traced program: ``models_band`` replaces lr/e1/e2 with literals and n_val
    # with ``n_frozen`` whenever the member is a star, so d(residual)/d(p[2:6])
    # is EXACTLY zero there -- and ``class_host_arrays`` sets
    # keep[:, :, 2:5] = (is_ext & mmask) and keep[:, :, 5] = fn_flag, both
    # all-False for a star-only class. The pre-K3 code still pushed all 6N
    # basis tangents through ``jax.linearize`` and then zeroed 4N of the
    # resulting rows/columns, i.e. it rendered 3x the tangent stamps it used,
    # formed a 3x-wider J J^T, and solved a 6N system whose dead block is the
    # identity.
    #
    # For a star-only class the live set is STATIC (the two position params of
    # every member slot), so the Jacobian is built on the live sub-basis and
    # the (nL, nL) / (nL,) results are scattered back into the full (nP, nP) /
    # (nP,) layout ONCE per hg_scan call. Everything downstream -- the keep
    # masking, the prior diagonal, the damping, the f64 solve, final_errors,
    # diverged/plateau probes -- is untouched and still sees the full-size H/g,
    # whose dead entries are exactly the zeros they were before.
    #
    # Exact-arithmetic identity, fp-noise in practice: the live block is now a
    # (nL, B^2) x (B^2, nL) GEMM instead of a sub-block of a (nP, B^2) one, so
    # cuBLAS may tile/accumulate it differently. Mixed classes and the
    # crossband GLS path keep the pre-K3 program verbatim.
    _k3 = (render_kind == "star") and (crossband is None) and N > 0
    if _k3:
        _live_np = np.concatenate(
            [np.array([6 * k, 6 * k + 1], dtype=np.int32) for k in range(N)])
    else:
        _live_np = np.arange(nP, dtype=np.int32)
    nL = int(_live_np.size)
    _reduced = bool(_k3 and nL < nP)
    # Preserve the per-lax.map-chunk TANGENT-LANE working set: the largest
    # divisor of nL that does not exceed the class's tangent_chunk (lax.map
    # needs an exact split; nL is 2 x a power of two here).
    if _reduced:
        tangent_chunk_live = max(
            d for d in range(1, min(int(tangent_chunk), nL) + 1) if nL % d == 0)
    else:
        tangent_chunk_live = int(tangent_chunk)

    def _psf_ns(psf_b):
        """Traced-array PSF view for ONE band (leaves already sliced;
        compatible with evaluate_psf_at / render_source_stamp_at:
        static H/W, traced coefs+origins)."""
        return types.SimpleNamespace(
            coefs=psf_b["coefs"],
            x_origin=psf_b["x_origin"], x_scale=psf_b["x_scale"],
            y_origin=psf_b["y_origin"], y_scale=psf_b["y_scale"],
            H_psf=T, W_psf=T,
        )

    def _render_star_cheap(psf_b, x, y, x0b, y0b):
        """Cheap (T,T) PSF stamp: raw PSF eval + Lanczos shift, NO galaxy
        FFT. Byte-identical to render_source_stamp_at's STAR output at
        tier T (same evaluate_psf_at at (x,y), same dx=x-x0b-round(.),
        same lanczos_shift_image) -- the star output there is always
        raw_psf (the discarded galaxy FFT is skipped here). Shared by the
        star-only kernel and the PERF #14 mixed-kernel cheap pass."""
        from .psf_polynomial import evaluate_psf_at
        from ..optimize.gpu.kernels.lanczos import lanczos_shift_image
        psf_jax = _psf_ns(psf_b)
        raw = evaluate_psf_at(psf_jax, x, y)
        x_grid = x - x0b
        y_grid = y - y0b
        dx = x_grid - jnp.round(x_grid)
        dy = y_grid - jnp.round(y_grid)
        return lanczos_shift_image(raw, dx, dy)

    def render_one(psf_b, x, y, x0b, y0b, stype, lr, e1, e2, n_val):
        """One member's (T,T) stamp in one band."""
        if render_kind == "star":
            # star-only fast path (no galaxy FFT).
            return _render_star_cheap(psf_b, x, y, x0b, y0b)
        psf_jax = _psf_ns(psf_b)
        return render_source_stamp_at(
            psf_jax, x, y, x0b, y0b, stype, lr, e1, e2,
            sersic_n=n_val, render_backend=render_backend,
        )

    def place(stamp, ix_top, iy_top):
        """(T,T) stamp -> (B,B) bbox window (stamp_top_left convention)."""
        py = jnp.arange(B)[:, None]
        px = jnp.arange(B)[None, :]
        sy = py - iy_top
        sx = px - ix_top
        inb = (sy >= 0) & (sy < T) & (sx >= 0) & (sx < T)
        syc = jnp.clip(sy, 0, T - 1)
        sxc = jnp.clip(sx, 0, T - 1)
        return jnp.where(inb, stamp[syc, sxc], 0.0)

    def models_band(p, isl, psf_b):
        """Per-member unit-flux models on the island bbox, ONE band.

        p: (N, 6). Returns (N, B, B), padded member slots zeroed via
        the member mask. Called inside the band lax.scan of the H/g
        accumulation, the chi2 evaluation, and the final flux solve
        (P5 J-level per-band restructure): the traced program holds
        ONE render and the per-island working set never scales with
        n_bands.
        """
        x, y = p[:, 0], p[:, 1]
        stype = isl["stype"]                       # (N,)
        is_star = stype == SOURCE_STAR
        # render-safe shapes for stars (production placeholders).
        lr = jnp.where(is_star, jnp.float32(0.0), p[:, 2])
        e1 = jnp.where(is_star, jnp.float32(1.0), p[:, 3])
        e2 = jnp.where(is_star, jnp.float32(0.0), p[:, 4])
        # ROUND-GALAXY SINGULARITY GUARD (P3b root-cause fix): the
        # ellipse polar decomposition (hypot/arctan2 in sersic_mog.
        # _ellipse_cov_terms and the direct-backend equivalent) has a
        # 0/0 JVP at EXACTLY ee1=ee2=0 -- one perfectly-round member
        # turns the whole island's forward-mode Jacobian to NaN, every
        # LM step is rejected, and the island silently burns max_steps
        # (the T08285 P3 perf sweep's "every mixed island unconverged"
        # was THIS, seeded by the drivers' ee=0 round injections; real
        # catalogs can carry ELL=0 rows too). Nudge exact zeros to a
        # dynamically-invisible ellipticity (axis ratio 0.9999).
        round0 = (~is_star) & (e1 == 0.0) & (e2 == 0.0)
        e1 = jnp.where(round0, jnp.float32(1e-4), e1)
        n_val = jnp.where(isl["free_n"], sersic_u_to_n(p[:, 5]),
                          isl["n_frozen"])
        x0b, y0b = isl["x0"], isl["y0"]            # scalars
        ix_top = jnp.round(x - x0b).astype(jnp.int32) - T // 2   # (N,)
        iy_top = jnp.round(y - y0b).astype(jnp.int32) - T // 2
        mm = isl["mmask"].astype(jnp.float32)      # (N,)

        if render_kind != "mixed" or not star_render_split:
            # star-only class, OR the split disabled (star_render_split=
            # False): single N-wide render. For a mixed class this is the
            # OLD pre-#14 path -- every lane (stars included) rendered via
            # render_source_stamp_at, so it is byte-identical to pre-#14.
            stamps = jax.vmap(
                lambda xk, yk, tk, lk, e1k, e2k, nk: render_one(
                    psf_b, xk, yk, x0b, y0b, tk, lk, e1k, e2k, nk)
            )(x, y, stype, lr, e1, e2, n_val)          # (N, T, T)
            placed = jax.vmap(place)(stamps, ix_top, iy_top)  # (N, B, B)
            return placed * mm[:, None, None]

        # ---- PERF #14 star/galaxy render split (fp-tolerance flavor) ----
        # A-tierT: the placed stamp is byte-identical PER ISLAND (star
        # slots == raw_psf, galaxy slots == the same table-MoG FFT), but
        # rendering the galaxies in a width-G vmap instead of width-N
        # shifts XLA's batched-FFT reduction order at the ulp level --
        # the SAME chunk-width fp-noise class this solver already carries
        # (module docstring); NOT bit-exact vs star_render_split=False,
        # but < 1 sigma with no systematic bias. Kills the wasted 512-pad
        # FFT that every STAR lane paid in a mixed island.
        # (a) CHEAP PSF pass over ALL N lanes (galaxy lanes overwritten).
        cheap = jax.vmap(
            lambda xk, yk: _render_star_cheap(psf_b, xk, yk, x0b, y0b)
        )(x, y)                                     # (N, T, T)
        placed_psf = jax.vmap(place)(cheap, ix_top, iy_top)   # (N, B, B)
        # (b) FFT pass over ONLY the G galaxy lanes (gathered at gsel from
        #     their ORIGINAL slots -- NO member reordering). Identical
        #     per-lane inputs (same p, round0 guard, n_val, x0b, ix_top).
        gsel = isl["gsel"]                          # (G,) int32 slot idx
        gvalid = isl["gvalid"]                      # (G,) bool
        gal = jax.vmap(
            lambda xk, yk, tk, lk, e1k, e2k, nk: render_source_stamp_at(
                _psf_ns(psf_b), xk, yk, x0b, y0b, tk, lk, e1k, e2k,
                sersic_n=nk, render_backend=render_backend)
        )(x[gsel], y[gsel], stype[gsel], lr[gsel], e1[gsel], e2[gsel],
          n_val[gsel])                              # (G, T, T)
        placed_gal = jax.vmap(place)(
            gal, ix_top[gsel], iy_top[gsel])        # (G, B, B)
        # (c) SCATTER galaxy renders into their ORIGINAL slots. gsel holds
        #     DISTINCT slot indices per island (galaxies first, padding on
        #     non-galaxy slots), so scatter-set is unique-index/
        #     deterministic; padding entries write the cheap PSF render back
        #     to a non-galaxy slot (a no-op). Star slots keep placed_psf.
        placed = placed_psf.at[gsel].set(
            jnp.where(gvalid[:, None, None], placed_gal, placed_psf[gsel]))
        return placed * mm[:, None, None]

    def _band_xs(isl, psf):
        """lax.scan xs over bands: per-band data/weights + the PSF tree
        (every leaf carries a leading n_bands axis, so scan slices one
        band per iteration)."""
        return (isl["data"], isl["w"], psf)

    def _fluxes_band(m_b, d_b, w_b, mm):
        """Joint VarPro flux solve for ONE band. m_b: (N, B, B).

        Returns (f, A64): fluxes (N,) float32 and the float64 normal
        matrix (N, N) for error estimates. Padded member slots -> zero
        rows/cols + unit diagonal (PD ridge).

        (Catalog-driven per-band FLUX priors are DEFERRED to the future
        field-level flux solve / P7 de-gridding; the reported flux is
        the data-only pass-2 S1' solve, so a flux prior on this discarded
        internal island flux would not affect any reported value.)
        """
        wm = w_b[None] * m_b                       # (N, B, B)
        A = jnp.einsum("kij,lij->kl", wm, m_b)     # (N, N)
        bvec = jnp.einsum("kij,ij->k", wm, d_b)    # (N,)
        ridge = jnp.where(mm, reg32, 1.0)
        A64 = A.astype(jnp.float64) + jnp.diag(ridge).astype(jnp.float64)
        b64 = bvec.astype(jnp.float64)
        f = jnp.linalg.solve(A64, b64[..., None])[..., 0]
        f = f.astype(jnp.float32) * mm
        return f, A64

    def residual_band(p_flat, isl, d_b, w_b, psf_b):
        """Whitened ONE-BAND residual with closed-form joint member
        fluxes. The VarPro flux solve is per-band by construction, so
        the band split is mathematically exact -- the full residual is
        the concatenation of these over bands. Always (B*B,) -- the
        NONLINEAR (position/shape/n) priors act OUTSIDE this residual
        (chi2_of penalty + one_step/final_errors Hessian diagonal), and
        flux priors are deferred (P7), so no rows are appended here."""
        p = p_flat.reshape(N, 6)
        m_b = models_band(p, isl, psf_b)
        f_b, _ = _fluxes_band(m_b, d_b, w_b, isl["mmask"])
        model_img = jnp.einsum("k,kij->ij", f_b, m_b)
        r = jnp.sqrt(w_b) * (d_b - model_img)
        return r.reshape(-1)                       # (B*B,)

    # ---- F3-S1: rank-r cross-band GLS machinery (crossband != None) ----
    # Identity (Woodbury): with FLAT sky weights (D = I exactly -- the
    # crossband stage applies NO per-band whitening) and ambient
    # covariance sum_i lam_i (u_i o p~_b(k)) (.)^H, the GLS chi2 equals
    # the existing per-band chi2 MINUS the filtered power of r
    # ambient-colored matched-filter channels of the residual:
    #   z_i(k) = sqrt(lam_i) sum_b u_ib conj(q_b(k)) r^_b(k),
    #   M(k)   = I_r + (sqrt(lam_i lam_j) sum_b u_ib u_jb |q_b(k)|^2)_ij,
    #   chi2  -= (1/B^2) sum_fullk z^H M^-1 z   (rfft double-count wk).
    # q_b = FFT of the band PSF at the island reference position (member-
    # mean); NO per-channel model is rendered -> data/model filter
    # identity holds BY CONSTRUCTION. Placement phases cancel because the
    # tier T (and hence stamp placement) is shared across bands. The
    # dq/dp derivative is neglected in hg_scan (PSF polynomial varies
    # slowly; quasi-Newton approximation on the objective only). The
    # sub-floor eigen-tail (lam < 1 sky unit) is absorbed into D=I
    # (disclosed; that is the constant-free rank rule's meaning).
    if crossband is not None:
        from .psf_polynomial import evaluate_psf_at as _cb_eval_psf
        cb_r = int(crossband["rank"])
        _cbU_T = jnp.asarray(np.asarray(
            crossband["u"], np.float32).T)             # (n_bands, r)
        _cbSL = jnp.asarray(np.sqrt(np.clip(np.asarray(
            crossband["lam"], np.float64), 0.0, None)).astype(np.float32))
        _nK = B * (B // 2 + 1)
        _wk_np = np.full((B, B // 2 + 1), 2.0, np.float32)
        _wk_np[:, 0] = 1.0
        if B % 2 == 0:
            _wk_np[:, -1] = 1.0
        _cbWK = jnp.asarray(_wk_np.reshape(-1))
        _invB2 = jnp.float32(1.0 / float(B * B))
        _cbEye = jnp.eye(cb_r, dtype=jnp.complex64)
        if T <= B:
            def _cb_stampB(s):
                return jnp.zeros((B, B), s.dtype).at[:T, :T].set(s)
        else:
            _cb_off = (T - B) // 2

            def _cb_stampB(s):
                return s[_cb_off:_cb_off + B, _cb_off:_cb_off + B]

        def _cb_band_xs(isl, psf):
            return (isl["data"], isl["w"], psf, _cbU_T)

        def _cb_ref_xy(p_flat, isl):
            p = p_flat.reshape(N, 6)
            mm = isl["mmask"].astype(jnp.float32)
            den = jnp.maximum(mm.sum(), 1.0)
            return (jnp.sum(p[:, 0] * mm) / den,
                    jnp.sum(p[:, 1] * mm) / den)

        def _cb_q(psf_b, cx, cy):
            stamp = _cb_eval_psf(_psf_ns(psf_b), cx, cy)
            return jnp.fft.rfft2(
                _cb_stampB(stamp).astype(jnp.float32)).reshape(-1)

        def _cb_Mk(Mc):
            return (Mc.transpose(2, 0, 1).astype(jnp.complex64)
                    + _cbEye[None])

    if crossband is None:
        def chi2_of(p_flat, isl, psf):
            def body(c, xs):
                d_b, w_b, psf_b = xs
                r = residual_band(p_flat, isl, d_b, w_b, psf_b)
                return c + jnp.sum(r * r), None
            c, _ = jax.lax.scan(body, jnp.float32(0.0), _band_xs(isl, psf))
            if has_priors:
                # nonlinear (position/shape/n) prior penalty (the exact
                # integral of the g/H prior terms folded in one_step).
                keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
                pen = jnp.sum(jnp.where(
                    keep,
                    (isl["prior_inv_sigma"]
                     * (p_flat - isl["prior_mean"])) ** 2,
                    0.0))
                c = c + pen
            return c
    else:
        def chi2_of(p_flat, isl, psf):
            cx, cy = _cb_ref_xy(p_flat, isl)

            def body(carry, xs):
                c, zc, Mc = carry
                d_b, w_b, psf_b, u_b = xs
                r = residual_band(p_flat, isl, d_b, w_b, psf_b)
                qf = _cb_q(psf_b, cx, cy)
                rh = jnp.fft.rfft2(r.reshape(B, B)).reshape(-1)
                su = (_cbSL * jnp.sqrt(jnp.maximum(
                    isl["cb_scale"], 0.0))) * u_b
                zc = zc + su[:, None] * (jnp.conj(qf) * rh)[None, :]
                q2 = qf.real ** 2 + qf.imag ** 2
                Mc = Mc + (su[:, None, None] * su[None, :, None]
                           * q2[None, None, :])
                return (c + jnp.sum(r * r), zc, Mc), None

            (c, zc, Mc), _ = jax.lax.scan(
                body,
                (jnp.float32(0.0),
                 jnp.zeros((cb_r, _nK), jnp.complex64),
                 jnp.zeros((cb_r, cb_r, _nK), jnp.float32)),
                _cb_band_xs(isl, psf))
            Mk = _cb_Mk(Mc)
            zk = zc.T
            sol = jnp.linalg.solve(Mk, zk[..., None])[..., 0]
            corr = jnp.sum(_cbWK * jnp.real(
                jnp.sum(jnp.conj(zk) * sol, axis=-1))) * _invB2
            c = c - corr
            if has_priors:
                keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
                pen = jnp.sum(jnp.where(
                    keep,
                    (isl["prior_inv_sigma"]
                     * (p_flat - isl["prior_mean"])) ** 2,
                    0.0))
                c = c + pen
            return c

    # dlnp mode: float64-CARRY chi2 (E0-validated arithmetic, dlnp ledger
    # analysis_dlnp/e0). IDENTICAL residual math (f32 renders/whitening)
    # with the per-pixel square + the accumulation carried in float64; in
    # crossband mode the z/M channels and the complex solve stay
    # complex64/f32 exactly as chi2_of -- only the per-band r.r sums and
    # the final k-sum of the Woodbury correction are f64 (carry-precision
    # semantics, not a math change). Removes the measured fp32
    # eval-noise-vs-quantum collision (plain: noise = 1.0x ulp32 -> quantum
    # unresolvable at chi2 >~ 1e7; crossband GLS corr-cancellation
    # amplifies to 2.6-5.7x ulp -> coin-flip accepts from chi2 ~ 1e6 and
    # the 41%-wrong-reject Pool-2 stall mechanism).
    chi2_of64 = None
    # The 1-sigma statistical quantum (dchi2 = 2*dlnp; 2*0.5 = 1 by
    # default) -- the SAME constant the shipped plateau certificate uses,
    # entering the step program only in dlnp mode.
    _QUANTUM64 = jnp.float64(2.0 * float(plateau_dlnp_tol))
    if convergence == "dlnp" and crossband is None:
        def chi2_of64(p_flat, isl, psf):
            def body(c, xs):
                d_b, w_b, psf_b = xs
                r = residual_band(p_flat, isl, d_b, w_b, psf_b)
                r64 = r.astype(jnp.float64)
                return c + jnp.sum(r64 * r64), None
            c, _ = jax.lax.scan(body, jnp.float64(0.0), _band_xs(isl, psf))
            if has_priors:
                keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
                pen = jnp.sum(jnp.where(
                    keep,
                    ((isl["prior_inv_sigma"]
                      * (p_flat - isl["prior_mean"])) ** 2
                     ).astype(jnp.float64),
                    0.0))
                c = c + pen
            return c
    elif convergence == "dlnp":
        def chi2_of64(p_flat, isl, psf):
            cx, cy = _cb_ref_xy(p_flat, isl)

            def body(carry, xs):
                c, zc, Mc = carry
                d_b, w_b, psf_b, u_b = xs
                r = residual_band(p_flat, isl, d_b, w_b, psf_b)
                qf = _cb_q(psf_b, cx, cy)
                rh = jnp.fft.rfft2(r.reshape(B, B)).reshape(-1)
                su = (_cbSL * jnp.sqrt(jnp.maximum(
                    isl["cb_scale"], 0.0))) * u_b
                zc = zc + su[:, None] * (jnp.conj(qf) * rh)[None, :]
                q2 = qf.real ** 2 + qf.imag ** 2
                Mc = Mc + (su[:, None, None] * su[None, :, None]
                           * q2[None, None, :])
                r64 = r.astype(jnp.float64)
                return (c + jnp.sum(r64 * r64), zc, Mc), None

            (c, zc, Mc), _ = jax.lax.scan(
                body,
                (jnp.float64(0.0),
                 jnp.zeros((cb_r, _nK), jnp.complex64),
                 jnp.zeros((cb_r, cb_r, _nK), jnp.float32)),
                _cb_band_xs(isl, psf))
            Mk = _cb_Mk(Mc)
            zk = zc.T
            sol = jnp.linalg.solve(Mk, zk[..., None])[..., 0]
            corr_k = _cbWK * jnp.real(
                jnp.sum(jnp.conj(zk) * sol, axis=-1))
            corr = jnp.sum(corr_k.astype(jnp.float64)) * jnp.float64(
                1.0 / float(B * B))
            c = c - corr
            if has_priors:
                keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
                pen = jnp.sum(jnp.where(
                    keep,
                    ((isl["prior_inv_sigma"]
                      * (p_flat - isl["prior_mean"])) ** 2
                     ).astype(jnp.float64),
                    0.0))
                c = c + pen
            return c

    # per-param step clips, tiled over members
    clip_vec = np.tile(
        np.asarray([pos_step_max, pos_step_max, shape_step_max,
                    shape_step_max, shape_step_max, un_step_max],
                   dtype=np.float32), N)
    # static mask: True for every param slot EXCEPT u_n (index 5 mod 6)
    _not_un = np.tile(
        np.asarray([True, True, True, True, True, False]), N)

    def hg_scan(p_flat, isl, psf):
        """UN-masked, UN-damped flux-marginalized GN (H, g) at p_flat.

        J-LEVEL PER-BAND ACCUMULATION (P5 speed fix b): scan over
        bands, computing the one-band Jacobian J_b (nP x B^2) and
        accumulating H += J_b J_b^T and g += J_b r_b in the carry.
        The VarPro flux solve is already per-band, so the band split
        is exact -- only the fp reduction order over bands changes
        (results match the all-band J to float tolerance, not bit).
        This removes the (nP x n_bands*B^2) J and the all-band
        tangent lanes from the per-island working set (~n_bands-fold
        smaller), which is what lets hard mixed buckets run at 4-8x
        larger island chunks. Tangents stay chunked inside the band
        (rule 3); lax.map keeps ONE compiled segment regardless of
        chunk count (a python unroll would explode compile time at
        nP=384).

        SHARED by ``one_step`` (the LM step) and ``final_errors`` (the
        post-convergence #28a covariance) so the traced ops are
        identical by construction -- the step path's jaxpr cannot
        drift from the error path's.
        """
        # K3: live sub-basis for star-only classes (identical rows of eye(nP),
        # just the ones whose Jacobian column is not identically zero).
        basis = jnp.eye(nP, dtype=p_flat.dtype)[jnp.asarray(_live_np)] \
            if _reduced else jnp.eye(nP, dtype=p_flat.dtype)
        n_tc = nL // tangent_chunk_live   # exact split by construction

        if crossband is None:
            def hg_band(carry, xs):
                Hc, gc = carry
                d_b, w_b, psf_b = xs
                rfun = lambda q: residual_band(q, isl, d_b, w_b, psf_b)  # noqa: E731
                r_b, lin = jax.linearize(rfun, p_flat)
                if n_tc <= 1:
                    J_b = jax.vmap(lin)(basis)
                else:
                    J_b = jax.lax.map(
                        lambda bc: jax.vmap(lin)(bc),
                        basis.reshape(n_tc, tangent_chunk_live, nP),
                    ).reshape(nL, -1)                  # (nL, B*B)
                return (Hc + J_b @ J_b.T, gc + J_b @ r_b), None

            (H, g), _ = jax.lax.scan(
                hg_band,
                (jnp.zeros((nL, nL), dtype=p_flat.dtype),
                 jnp.zeros((nL,), dtype=p_flat.dtype)),
                _band_xs(isl, psf))
            if _reduced:
                _li = jnp.asarray(_live_np)
                H = jnp.zeros((nP, nP), dtype=H.dtype).at[
                    _li[:, None], _li[None, :]].set(H)
                g = jnp.zeros((nP,), dtype=g.dtype).at[_li].set(g)
            return H, g

        # F3-S1 crossband: same per-band J accumulation + the z/M/JZ
        # channel carries; after the scan subtract the Woodbury GLS
        # correction from H and g (same half-derivative convention as
        # H = J J^T, g = J r). M is real-symmetric (built from |q|^2).
        # crossband: pre-K3 program verbatim (full 6N basis, nP-sized carry).
        n_tc = nP // tangent_chunk
        cx, cy = _cb_ref_xy(p_flat, isl)

        def hg_band(carry, xs):
            Hc, gc, zc, Mc, JZc = carry
            d_b, w_b, psf_b, u_b = xs
            rfun = lambda q: residual_band(q, isl, d_b, w_b, psf_b)  # noqa: E731
            r_b, lin = jax.linearize(rfun, p_flat)
            if n_tc <= 1:
                J_b = jax.vmap(lin)(basis)
            else:
                J_b = jax.lax.map(
                    lambda bc: jax.vmap(lin)(bc),
                    basis.reshape(n_tc, tangent_chunk, nP),
                ).reshape(nP, -1)                  # (nP, B*B)
            qf = _cb_q(psf_b, cx, cy)
            rh = jnp.fft.rfft2(r_b.reshape(B, B)).reshape(-1)
            su = (_cbSL * jnp.sqrt(jnp.maximum(
                isl["cb_scale"], 0.0))) * u_b
            zc = zc + su[:, None] * (jnp.conj(qf) * rh)[None, :]
            q2 = qf.real ** 2 + qf.imag ** 2
            Mc = Mc + (su[:, None, None] * su[None, :, None]
                       * q2[None, None, :])
            Jh = jnp.fft.rfft2(J_b.reshape(nP, B, B)).reshape(nP, -1)
            JZc = JZc + (su[None, :, None]
                         * (jnp.conj(qf)[None, None, :]
                            * Jh[:, None, :]))
            return (Hc + J_b @ J_b.T, gc + J_b @ r_b, zc, Mc, JZc), None

        (H, g, zc, Mc, JZc), _ = jax.lax.scan(
            hg_band,
            (jnp.zeros((nP, nP), dtype=p_flat.dtype),
             jnp.zeros((nP,), dtype=p_flat.dtype),
             jnp.zeros((cb_r, _nK), jnp.complex64),
             jnp.zeros((cb_r, cb_r, _nK), jnp.float32),
             jnp.zeros((nP, cb_r, _nK), jnp.complex64)),
            _cb_band_xs(isl, psf))
        Mk = _cb_Mk(Mc)                             # (nK, r, r)
        zk = zc.T                                    # (nK, r)
        solz = jnp.linalg.solve(Mk, zk[..., None])[..., 0]
        JZk = JZc.transpose(2, 0, 1)                 # (nK, nP, r)
        solJ = jnp.linalg.solve(
            Mk, JZk.conj().transpose(0, 2, 1))       # (nK, r, nP)
        Hc2 = jnp.real(jnp.einsum(
            'k,kpr,krq->pq', _cbWK, JZk, solJ)) * _invB2
        gc2 = jnp.real(jnp.einsum(
            'k,kpr,kr->p', _cbWK, jnp.conj(JZk), solz)) * _invB2
        return H - Hc2.astype(H.dtype), g - gc2.astype(g.dtype)

    def one_step(p_flat, lam, chi2_cur, consec, nrej, conv, stalled,
                 nsteps, isl, psf):
        """One joint LM step for one island (frozen when conv). LEGACY
        stall counting (every rejected step counts). #35 note: this
        program is BYTE-IDENTICAL to the pre-#35 step -- the diverged
        disclosure is reconstructed POST-HOC by ``diverged_probe`` on
        the stalled islands' final state (measured: ANY change to this
        program, even aux-only outputs of existing tensors, shifts XLA
        fusion/autotuning at the ulp level and flips stall-marginal
        trajectories -- the documented chunk-width fp-noise class)."""
        H, g = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)  # (nP,)
        # u_n ANNEALING (P3b): free-n islands slide along the n-Re
        # degeneracy ridge with dchi2 > ftol_rel*chi2 for 30+ steps at
        # high S/N (measured on T08285 injections: chi2 keeps creeping,
        # every step accepted, no plateau). Production freezes n below
        # an S/N gate for the same reason; here n gets un_freeze_steps
        # LM steps (plenty: n moves fastest early, and its remaining
        # drift trades against Re at the noise level), then freezes so
        # the plateau detectors can fire on the well-conditioned rest.
        if un_freeze_steps > 0:
            keep = keep & (jnp.asarray(_not_un) | (nsteps < un_freeze_steps))
        keep_o = keep[:, None] & keep[None, :]
        H = jnp.where(keep_o, H, 0.0) + jnp.diag(jnp.where(keep, 0.0, 1.0))
        g = jnp.where(keep, g, 0.0)
        if has_priors:
            # Catalog-driven Gaussian prior on the nonlinear params
            # (mean = seed). Compute the LM damping from the PURE GN
            # diagonal FIRST, then add the prior diagonal UNDAMPED so
            # lam scales only the data-Hessian (the prior is not damped
            # away). Keep-masked; inv_sigma == 0 is an exact no-op.
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
            prior_diag = jnp.where(keep, prior_inv2, 0.0)
            g = g + jnp.where(
                keep, prior_inv2 * (p_flat - isl["prior_mean"]), 0.0)
            H_d = (H + lam * jnp.diag(jnp.diag(H)) + jnp.diag(prior_diag)
                   + 1e-8 * jnp.eye(nP, dtype=H.dtype))
        else:
            H_d = (H + lam * jnp.diag(jnp.diag(H))
                   + 1e-8 * jnp.eye(nP, dtype=H.dtype))
        # float64 solve, like production's linear steps
        delta = -jnp.linalg.solve(
            H_d.astype(jnp.float64), g.astype(jnp.float64)
        ).astype(p_flat.dtype)
        delta = jnp.where(keep, delta, 0.0)
        delta = jnp.clip(delta, -jnp.asarray(clip_vec), jnp.asarray(clip_vec))
        p_try = jnp.clip(p_flat + delta, isl["lower"], isl["upper"])
        chi2_try = chi2_of(p_try, isl, psf)

        accept = chi2_try < chi2_cur
        p_new = jnp.where(accept, p_try, p_flat)
        lam_new = jnp.where(accept, jnp.maximum(lam * LM_DOWN, 1e-9),
                            jnp.minimum(lam * LM_UP, 1e6))
        chi2_new = jnp.where(accept, chi2_try, chi2_cur)
        dchi2 = chi2_cur - chi2_try
        step_eff = (p_new - p_flat).reshape(N, 6)
        mmf = isl["mmask"].astype(jnp.float32)
        pos_step = jnp.max(mmf * jnp.hypot(step_eff[:, 0], step_eff[:, 1]))
        # per-param shape/u_n steps (P3b prong 2): max |d| over members
        shape_step = jnp.max(mmf[:, None] * jnp.abs(step_eff[:, 2:6]))
        # prong 1: RELATIVE ftol on 2 consecutive accepted steps
        small_dchi = accept & (dchi2 < ftol_rel * jnp.maximum(chi2_cur, 1.0))
        consec_new = jnp.where(small_dchi, consec + 1,
                               jnp.where(accept, 0, consec))
        # prong 2: all param steps below per-param tolerances
        step_small = accept & (pos_step < step_tol) \
            & (shape_step < shape_step_tol)
        # prong 3: k consecutive rejected steps -> converged-stalled
        nrej_new = jnp.where(accept, 0, nrej + 1)
        stall_now = nrej_new >= STALL_REJECT_K
        conv_new = conv | (consec_new >= 2) | step_small | stall_now

        # frozen islands pass state through untouched
        p_out = jnp.where(conv, p_flat, p_new)
        lam_out = jnp.where(conv, lam, lam_new)
        chi2_out = jnp.where(conv, chi2_cur, chi2_new)
        consec_out = jnp.where(conv, consec, consec_new)
        nrej_out = jnp.where(conv, nrej, nrej_new)
        stalled_out = jnp.where(conv, stalled, stall_now)
        nsteps_out = jnp.where(conv, nsteps, nsteps + 1)
        return (p_out, lam_out, chi2_out, consec_out, nrej_out,
                conv_new, stalled_out, nsteps_out)

    def one_step_tr(p_flat, delta_r, chi2_cur, consec, nrej, conv, stalled,
                    nsteps, isl, psf):
        """Trust-region (scaled dogleg) variant of one_step (step_mode=
        "trust_region", OPT-IN; off => one_step, bit-identical). Replaces the
        fixed per-param clip + lam-only damping escalation + the 4-reject
        clip-saturated-frozen-proposal stall with a parameter-free trust radius
        Delta (carried in the legacy `lam`/`delta_r` slot) updated from the chi2
        gain ratio rho (Nocedal-Wright Alg 4.1 / More 1978 constants --
        algorithm-standard, NOT dataset-tuned). ACCEPT semantics (chi2 must
        decrease) and the three convergence prongs are UNCHANGED; only the step
        geometry (scaled dogleg vs damped-Newton-then-clip) and the radius
        update (vs lam x10/x0.1) differ. A rejected step now SHRINKS Delta, so
        the K stall-rejects test progressively smaller steps (a genuine
        plateau), not one frozen clip-saturated proposal. Robustness /
        determinism / stall-conversion lever -- NOT an accuracy fix: on a
        displaced-min valley it reaches the (wrong) min faster; disclose via
        railed_probe / island_railed (diverged_probe's clip-saturation
        signature is vacuous here since rejects change the proposal)."""
        H, g = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
        if un_freeze_steps > 0:
            keep = keep & (jnp.asarray(_not_un) | (nsteps < un_freeze_steps))
        keep_o = keep[:, None] & keep[None, :]
        H = jnp.where(keep_o, H, 0.0) + jnp.diag(jnp.where(keep, 0.0, 1.0))
        g = jnp.where(keep, g, 0.0)
        if has_priors:
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
            prior_diag = jnp.where(keep, prior_inv2, 0.0)
            g = g + jnp.where(
                keep, prior_inv2 * (p_flat - isl["prior_mean"]), 0.0)
            Hm = H + jnp.diag(prior_diag) + 1e-8 * jnp.eye(nP, dtype=H.dtype)
        else:
            Hm = H + 1e-8 * jnp.eye(nP, dtype=H.dtype)
        H64 = Hm.astype(jnp.float64)
        g64 = g.astype(jnp.float64)
        Dr = jnp.asarray(delta_r, jnp.float64)
        # Diagonal (Marquardt) scaling D = sqrt(diag H); frozen slots carry a
        # unit diagonal (line above) -> D=1, g=0 -> zero step, exact no-op.
        dsc = jnp.sqrt(jnp.maximum(jnp.diag(H64), 1e-12))
        # Gauss-Newton step (scaled: s = D.delta)
        p_gn = -jnp.linalg.solve(H64, g64)
        s_gn = dsc * p_gn
        n_gn = jnp.linalg.norm(s_gn)
        # First step (Delta initialised to 0 as an "unset" sentinel by
        # run_class_chunk): start the radius at the full GN step's scaled
        # magnitude -- a parameter-free Delta0 tied to today's first step.
        Dr = jnp.where(Dr <= 0.0, jnp.maximum(n_gn, 1e-6), Dr)
        # Cauchy step (scaled steepest descent): g_hat = D^-1 g,
        # H_hat = D^-1 H D^-1 ; s_C = -(g_hat.g_hat / g_hat.H_hat.g_hat) g_hat
        gh = g64 / dsc
        Hhat_gh = (H64 @ (gh / dsc)) / dsc
        ghHgh = jnp.dot(gh, Hhat_gh)
        gg = jnp.dot(gh, gh)
        alpha = jnp.where(ghHgh > 0.0, gg / jnp.maximum(ghHgh, 1e-30), 1.0)
        s_c = -alpha * gh
        n_c = jnp.linalg.norm(s_c)
        # Dogleg: GN if inside TR; scaled-Cauchy-to-boundary if even the Cauchy
        # point is outside; else blend s_c -> s_gn to the boundary ||s||=Delta.
        diff = s_gn - s_c
        aa = jnp.dot(diff, diff)
        bb = 2.0 * jnp.dot(s_c, diff)
        cc = jnp.dot(s_c, s_c) - Dr * Dr
        disc = jnp.maximum(bb * bb - 4.0 * aa * cc, 0.0)
        t_blend = jnp.clip(
            (-bb + jnp.sqrt(disc)) / jnp.maximum(2.0 * aa, 1e-30), 0.0, 1.0)
        s_blend = s_c + t_blend * diff
        s_cbnd = jnp.where(n_c > 0.0, (Dr / jnp.maximum(n_c, 1e-30)) * s_c, s_c)
        s = jnp.where(n_gn <= Dr, s_gn,
                      jnp.where(n_c >= Dr, s_cbnd, s_blend))
        s_norm = jnp.linalg.norm(s)
        delta = jnp.where(keep, (s / dsc).astype(p_flat.dtype), 0.0)
        p_try = jnp.clip(p_flat + delta, isl["lower"], isl["upper"])
        chi2_try = chi2_of(p_try, isl, psf)
        # gain ratio on the REALIZED (box-projected) step
        d_real = (p_try - p_flat).astype(jnp.float64)
        pred = -(jnp.dot(g64, d_real) + 0.5 * jnp.dot(d_real, H64 @ d_real))
        actual = (chi2_cur - chi2_try).astype(jnp.float64)
        rho = actual / jnp.where(pred > 0.0, pred, 1e-30)
        # radius update (NW Alg 4.1); the shrink branch also covers rejects
        Dmax = jnp.linalg.norm(
            dsc * (isl["upper"] - isl["lower"]).astype(jnp.float64))
        at_bnd = s_norm >= 0.99 * Dr
        Dr_new = jnp.where(
            rho < TR_ETA_LO, Dr * TR_SHRINK,
            jnp.where((rho > TR_ETA_HI) & at_bnd,
                      jnp.minimum(Dr * TR_EXPAND, Dmax), Dr))
        Dr_new = jnp.asarray(Dr_new, delta_r.dtype)
        # accept iff chi2 decreased -- identical to one_step
        accept = chi2_try < chi2_cur
        p_new = jnp.where(accept, p_try, p_flat)
        chi2_new = jnp.where(accept, chi2_try, chi2_cur)
        dchi2 = chi2_cur - chi2_try
        step_eff = (p_new - p_flat).reshape(N, 6)
        mmf = isl["mmask"].astype(jnp.float32)
        pos_step = jnp.max(mmf * jnp.hypot(step_eff[:, 0], step_eff[:, 1]))
        shape_step = jnp.max(mmf[:, None] * jnp.abs(step_eff[:, 2:6]))
        small_dchi = accept & (dchi2 < ftol_rel * jnp.maximum(chi2_cur, 1.0))
        consec_new = jnp.where(small_dchi, consec + 1,
                               jnp.where(accept, 0, consec))
        step_small = accept & (pos_step < step_tol) \
            & (shape_step < shape_step_tol)
        nrej_new = jnp.where(accept, 0, nrej + 1)
        stall_now = nrej_new >= STALL_REJECT_K
        conv_new = conv | (consec_new >= 2) | step_small | stall_now
        p_out = jnp.where(conv, p_flat, p_new)
        dr_out = jnp.where(conv, delta_r, Dr_new)
        chi2_out = jnp.where(conv, chi2_cur, chi2_new)
        consec_out = jnp.where(conv, consec, consec_new)
        nrej_out = jnp.where(conv, nrej, nrej_new)
        stalled_out = jnp.where(conv, stalled, stall_now)
        nsteps_out = jnp.where(conv, nsteps, nsteps + 1)
        return (p_out, dr_out, chi2_out, consec_out, nrej_out,
                conv_new, stalled_out, nsteps_out)

    def one_step_changed(p_flat, lam, chi2_cur, consec, nrej, conv,
                         stalled, diverged, n_same, prej, nsteps, isl,
                         psf):
        """#35 ``stall_count_mode="changed"`` step (OPT-IN, science-
        affecting by design -- no bit-compat claim vs the legacy step).

        A rejected step counts toward the STALL_REJECT_K stall ONLY when
        its clip/bound-limited proposal DIFFERS from the previously
        rejected one; identical-proposal rejects keep escalating lam
        without counting, so the solver eventually tests genuinely
        smaller steps and the stall means "uphill at K distinct
        dampings". The counting rule itself needs the proposal-identity
        signal, so here the tracker state (``diverged``/``n_same``/
        ``prej``) lives IN-KERNEL: ``prej`` = last rejected proposal,
        ``n_same`` = length of the current identical-proposal reject run
        (identity under lam escalation is only possible when every kept
        parameter is pinned at its per-step clip or bound, or lam is at
        its cap), ``diverged`` = sticky disclosure flag set once n_same
        reaches STALL_REJECT_K. Everything else mirrors the legacy
        step."""
        H, g = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)  # (nP,)
        # u_n annealing: see one_step.
        if un_freeze_steps > 0:
            keep = keep & (jnp.asarray(_not_un) | (nsteps < un_freeze_steps))
        keep_o = keep[:, None] & keep[None, :]
        H = jnp.where(keep_o, H, 0.0) + jnp.diag(jnp.where(keep, 0.0, 1.0))
        g = jnp.where(keep, g, 0.0)
        if has_priors:
            # see one_step (identical prior injection).
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
            prior_diag = jnp.where(keep, prior_inv2, 0.0)
            g = g + jnp.where(
                keep, prior_inv2 * (p_flat - isl["prior_mean"]), 0.0)
            H_d = (H + lam * jnp.diag(jnp.diag(H)) + jnp.diag(prior_diag)
                   + 1e-8 * jnp.eye(nP, dtype=H.dtype))
        else:
            H_d = (H + lam * jnp.diag(jnp.diag(H))
                   + 1e-8 * jnp.eye(nP, dtype=H.dtype))
        delta = -jnp.linalg.solve(
            H_d.astype(jnp.float64), g.astype(jnp.float64)
        ).astype(p_flat.dtype)
        delta = jnp.where(keep, delta, 0.0)
        delta = jnp.clip(delta, -jnp.asarray(clip_vec), jnp.asarray(clip_vec))
        p_try = jnp.clip(p_flat + delta, isl["lower"], isl["upper"])
        chi2_try = chi2_of(p_try, isl, psf)

        accept = chi2_try < chi2_cur
        p_new = jnp.where(accept, p_try, p_flat)
        lam_new = jnp.where(accept, jnp.maximum(lam * LM_DOWN, 1e-9),
                            jnp.minimum(lam * LM_UP, 1e6))
        chi2_new = jnp.where(accept, chi2_try, chi2_cur)
        dchi2 = chi2_cur - chi2_try
        step_eff = (p_new - p_flat).reshape(N, 6)
        mmf = isl["mmask"].astype(jnp.float32)
        pos_step = jnp.max(mmf * jnp.hypot(step_eff[:, 0], step_eff[:, 1]))
        shape_step = jnp.max(mmf[:, None] * jnp.abs(step_eff[:, 2:6]))
        small_dchi = accept & (dchi2 < ftol_rel * jnp.maximum(chi2_cur, 1.0))
        consec_new = jnp.where(small_dchi, consec + 1,
                               jnp.where(accept, 0, consec))
        step_small = accept & (pos_step < step_tol) \
            & (shape_step < shape_step_tol)
        # #35 proposal-identity (constant-free exact comparison): a
        # rejected proposal identical to the PREVIOUS rejected proposal
        # means the lam escalation did not test a new step.
        prop_same = (n_same > 0) & jnp.all(p_try == prej)
        n_same_new = jnp.where(accept, 0,
                               jnp.where(prop_same, n_same + 1, 1))
        prej_new = jnp.where(accept, prej, p_try)
        # prong 3, "changed" counting: identical-proposal rejects do NOT
        # count (they keep escalating lam via lam_new above).
        nrej_new = jnp.where(accept, 0,
                             jnp.where(prop_same, nrej, nrej + 1))
        stall_now = nrej_new >= STALL_REJECT_K
        diverged_new = diverged | (n_same_new >= STALL_REJECT_K)
        conv_new = conv | (consec_new >= 2) | step_small | stall_now

        # frozen islands pass state through untouched
        p_out = jnp.where(conv, p_flat, p_new)
        lam_out = jnp.where(conv, lam, lam_new)
        chi2_out = jnp.where(conv, chi2_cur, chi2_new)
        consec_out = jnp.where(conv, consec, consec_new)
        nrej_out = jnp.where(conv, nrej, nrej_new)
        stalled_out = jnp.where(conv, stalled, stall_now)
        diverged_out = jnp.where(conv, diverged, diverged_new)
        n_same_out = jnp.where(conv, n_same, n_same_new)
        prej_out = jnp.where(conv, prej, prej_new)
        nsteps_out = jnp.where(conv, nsteps, nsteps + 1)
        return (p_out, lam_out, chi2_out, consec_out, nrej_out,
                conv_new, stalled_out, diverged_out, n_same_out, prej_out,
                nsteps_out)

    def one_step_dlnp(p_flat, lam, chi2_cur, consec, nrej, conv,
                      stalled, diverged, n_same, prej, nsteps, isl,
                      psf):
        """dlnp-mode step (``convergence="dlnp"``, OPT-IN,
        science-affecting by design -- own truth campaign,
        internal dlnp design note §2 + E0 amendments).

        Same 11-state layout as ``one_step_changed`` (so the LM loop and
        the A->B requeue carry are shared) with the chi2 slot carried in
        FLOAT64. Differences from the legacy criteria, each E0-measured:

        - accept test + dchi2 in the f64 carry (``chi2_of64``): the f32
          accept test near the optimum was a measured coin flip on big
          islands (wrong-rejects = 41% of crossband rejects = the Pool-2
          stall mechanism).
        - prong 1 ABSOLUTE: accepted ``dchi2 < 2*plateau_dlnp_tol`` (the
          1-sigma statistical quantum, island-size independent; Tractor
          optimize-loop dlnp semantics) on 2 consecutive accepted steps.
          The relative prong stopped chi2>1e5 islands while they were
          still harvesting ~20 quanta per accepted step.
        - prong 2 QUANTUM-GATED: the fixed 0.1px/0.01 step tolerances
          fire only when the accepted step is ALSO sub-quantum (E0: 194
          bright-star step-exits left >10 quanta each, nrej=0 -- the
          DAOPHOT unscaled-tolerance problem, measured).
        - stall counting = distinct proposals (the "changed" rule);
          identical-proposal rejects keep escalating lam x10 without
          counting, so K counted rejects certify K materially different
          dampings found no improvement (any improving proposal is an
          accept by construction).
        - terminal no-information reject-exits: a REJECT tested at the
          lam cap, or a NO-MOVE proposal (the damped update rounds to
          zero against the f32 params; lam escalation only shrinks it)
          -- either way every subsequent proposal is provably
          bit-identical, so exit now instead of walking the ladder
          (E1a measured: ~12 wasted identical rejects per finished
          island on UDS without the no-move exit).

        Every reject-exit gets the POST-HOC ``plateau_probe``
        certificate (same f32 arithmetic as legacy mode -> one
        island_stall_dchi2/island_plateau meaning across modes);
        run_class_chunk maps certificate-passing exits to
        CONVERGED-ON-PLATEAU and keeps only certificate-failing ones
        island_stalled (E1a exit-state mapping)."""
        H, g = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)  # (nP,)
        # u_n annealing: see one_step.
        if un_freeze_steps > 0:
            keep = keep & (jnp.asarray(_not_un) | (nsteps < un_freeze_steps))
        keep_o = keep[:, None] & keep[None, :]
        H = jnp.where(keep_o, H, 0.0) + jnp.diag(jnp.where(keep, 0.0, 1.0))
        g = jnp.where(keep, g, 0.0)
        if has_priors:
            # see one_step (identical prior injection).
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
            prior_diag = jnp.where(keep, prior_inv2, 0.0)
            g = g + jnp.where(
                keep, prior_inv2 * (p_flat - isl["prior_mean"]), 0.0)
            H_d = (H + lam * jnp.diag(jnp.diag(H)) + jnp.diag(prior_diag)
                   + 1e-8 * jnp.eye(nP, dtype=H.dtype))
        else:
            H_d = (H + lam * jnp.diag(jnp.diag(H))
                   + 1e-8 * jnp.eye(nP, dtype=H.dtype))
        delta = -jnp.linalg.solve(
            H_d.astype(jnp.float64), g.astype(jnp.float64)
        ).astype(p_flat.dtype)
        delta = jnp.where(keep, delta, 0.0)
        delta = jnp.clip(delta, -jnp.asarray(clip_vec), jnp.asarray(clip_vec))
        p_try = jnp.clip(p_flat + delta, isl["lower"], isl["upper"])
        chi2_try = chi2_of64(p_try, isl, psf)

        accept = chi2_try < chi2_cur
        p_new = jnp.where(accept, p_try, p_flat)
        lam_new = jnp.where(accept, jnp.maximum(lam * LM_DOWN, 1e-9),
                            jnp.minimum(lam * LM_UP, 1e6))
        chi2_new = jnp.where(accept, chi2_try, chi2_cur)
        dchi2 = chi2_cur - chi2_try                    # float64
        step_eff = (p_new - p_flat).reshape(N, 6)
        mmf = isl["mmask"].astype(jnp.float32)
        pos_step = jnp.max(mmf * jnp.hypot(step_eff[:, 0], step_eff[:, 1]))
        shape_step = jnp.max(mmf[:, None] * jnp.abs(step_eff[:, 2:6]))
        sub_quantum = dchi2 < _QUANTUM64
        # prong 1: ABSOLUTE statistical ftol on 2 consecutive accepts
        small_dchi = accept & sub_quantum
        consec_new = jnp.where(small_dchi, consec + 1,
                               jnp.where(accept, 0, consec))
        # prong 2: step tolerances, quantum-gated
        step_small = accept & (pos_step < step_tol) \
            & (shape_step < shape_step_tol) & sub_quantum
        # proposal identity (exact comparison, constant-free)
        prop_same = (n_same > 0) & jnp.all(p_try == prej)
        n_same_new = jnp.where(accept, 0,
                               jnp.where(prop_same, n_same + 1, 1))
        prej_new = jnp.where(accept, prej, p_try)
        # prong 3: K DISTINCT rejected proposals, or a reject with no
        # further information reachable -- tested AT the lam cap, or a
        # NO-MOVE proposal (the damped update rounds to zero against the
        # f32 params: lam escalation only shrinks it further, so every
        # future proposal is bit-identical; E1a measured anatomy: without
        # this exit such islands walk ~12 identical rejects to the cap).
        # Either way the exit is a REJECT-EXIT whose meaning the post-hoc
        # ladder certificate decides in run_class_chunk (certificate
        # passes -> converged-on-plateau; fails -> honest stall).
        nrej_new = jnp.where(accept, 0,
                             jnp.where(prop_same, nrej, nrej + 1))
        no_move = jnp.all(p_try == p_flat)
        stall_now = (nrej_new >= STALL_REJECT_K) \
            | ((~accept) & ((lam >= jnp.float32(1e6)) | no_move))
        diverged_new = diverged | (n_same_new >= STALL_REJECT_K)
        conv_new = conv | (consec_new >= 2) | step_small | stall_now

        # frozen islands pass state through untouched
        p_out = jnp.where(conv, p_flat, p_new)
        lam_out = jnp.where(conv, lam, lam_new)
        chi2_out = jnp.where(conv, chi2_cur, chi2_new)
        consec_out = jnp.where(conv, consec, consec_new)
        nrej_out = jnp.where(conv, nrej, nrej_new)
        stalled_out = jnp.where(conv, stalled, stall_now)
        diverged_out = jnp.where(conv, diverged, diverged_new)
        n_same_out = jnp.where(conv, n_same, n_same_new)
        prej_out = jnp.where(conv, prej, prej_new)
        nsteps_out = jnp.where(conv, nsteps, nsteps + 1)
        return (p_out, lam_out, chi2_out, consec_out, nrej_out,
                conv_new, stalled_out, diverged_out, n_same_out, prej_out,
                nsteps_out)

    def diverged_probe(p_flat, lam_final, nsteps_final, isl, psf):
        """#35 POST-HOC divergence reconstruction for one STALLED island
        (legacy stall counting; read-only, runs AFTER the LM loop like
        ``source_metrics``/``final_errors`` so the step program stays
        byte-identical to the pre-#35 solver).

        A stall is exactly ``STALL_REJECT_K`` consecutive REJECTED steps:
        the parameters never moved during the run (rejects keep
        ``p_flat``) and lam was escalated x10 per reject, so the K
        tested proposals are a pure function of the FINAL state --
        ``lam_j = lam_final * 10**(j - K)`` for ``j = 0..K-1`` at the
        stalled ``p_flat``. This probe recomputes them with the exact
        ``one_step`` arithmetic (same hg_scan/masking/damping/clips/
        bounds) and returns True iff all K proposals are IDENTICAL
        (element-wise) -- the clip-saturated-runaway signature (identity
        under lam escalation is only possible when every kept parameter
        is pinned at its per-step clip or bound, or lam is at its cap).

        Approximations (documented): lam_j is reconstructed by float
        division (ulp-level noise vs the true forward products --
        immaterial: identical proposals match EXACTLY via clip/bound
        pinning, distinct ones differ far above ulp); if lam_final sits
        AT the 1e6 cap the pre-cap lam is unrecoverable and the no-cap
        reconstruction is used; with ``un_freeze_steps > 0`` (opt-in)
        the per-test annealing gate is replicated from
        ``nsteps_final``."""
        H, g = hg_scan(p_flat, isl, psf)
        keep0 = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)  # (nP,)

        def prop_at(j):
            lam_j = lam_final * jnp.float32(
                10.0 ** (j - STALL_REJECT_K))
            keep = keep0
            if un_freeze_steps > 0:
                ns_j = nsteps_final - STALL_REJECT_K + j
                keep = keep & (jnp.asarray(_not_un)
                               | (ns_j < un_freeze_steps))
            keep_o = keep[:, None] & keep[None, :]
            Hm = jnp.where(keep_o, H, 0.0) + jnp.diag(
                jnp.where(keep, 0.0, 1.0))
            gm = jnp.where(keep, g, 0.0)
            if has_priors:
                prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
                prior_diag = jnp.where(keep, prior_inv2, 0.0)
                gm = gm + jnp.where(
                    keep, prior_inv2 * (p_flat - isl["prior_mean"]), 0.0)
                H_d = (Hm + lam_j * jnp.diag(jnp.diag(Hm))
                       + jnp.diag(prior_diag)
                       + 1e-8 * jnp.eye(nP, dtype=Hm.dtype))
            else:
                H_d = (Hm + lam_j * jnp.diag(jnp.diag(Hm))
                       + 1e-8 * jnp.eye(nP, dtype=Hm.dtype))
            delta = -jnp.linalg.solve(
                H_d.astype(jnp.float64), gm.astype(jnp.float64)
            ).astype(p_flat.dtype)
            delta = jnp.where(keep, delta, 0.0)
            delta = jnp.clip(delta, -jnp.asarray(clip_vec),
                             jnp.asarray(clip_vec))
            return jnp.clip(p_flat + delta, isl["lower"], isl["upper"])

        p_try0 = prop_at(0)
        same = jnp.array(True)
        for j in range(1, STALL_REJECT_K):
            same = same & jnp.all(prop_at(j) == p_try0)
        return same

    def plateau_probe(p_flat, nsteps_final, isl, psf):
        """Stall-yield campaign (2026-07-27): post-hoc PROBABILITY-
        CONVERGENCE certificate for one STALLED island (read-only,
        Tractor-style dlnp semantics -- user-directed design).

        A stall is K consecutive rejected steps, but "no damped step
        improves chi2 by a statistically meaningful amount" is
        CONVERGENCE, not failure: the meaningful quantum is ABSOLUTE
        (dchi2 = 2*dlnp; dchi2 = 1 <=> a 1-sigma single-parameter
        move), unlike the size-dependent relative ftol prong. The
        historical K rejects can all be ONE clip-pinned proposal (the
        clip-neutered-lambda anatomy, module docstring) and never
        sample the small-step regime, so this probe sweeps the FULL
        decade lambda ladder 1e-9..1e6 -- including the large-lambda
        (small-step) rungs the reject budget never reaches -- with the
        exact one_step arithmetic (hg/priors/damping/clips/bounds) and
        returns  min_j chi2(prop(lam_j)) - chi2(p) : the best available
        damped-step improvement in chi2 units (negative = improvement
        exists). Emitted as island_stall_dchi2; the island_plateau bool
        (best > -2*island.plateau_dlnp_tol) is derived HOST-side so the
        tolerance never enters the kernel. Runs AFTER the LM loop on
        the stalled subset only, like diverged_probe -- the step
        program stays byte-identical. Mode-independent (needs no
        reject-history reconstruction). DISCLOSURE-ONLY at ship time:
        continuation-on-improvement is deliberately NOT built (measured
        trap: it descends displaced valleys deeper)."""
        chi2_cur = chi2_of(p_flat, isl, psf)
        H, g = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
        if un_freeze_steps > 0:
            keep = keep & (jnp.asarray(_not_un)
                           | (nsteps_final < un_freeze_steps))
        keep_o = keep[:, None] & keep[None, :]
        Hm = jnp.where(keep_o, H, 0.0) + jnp.diag(
            jnp.where(keep, 0.0, 1.0))
        gm = jnp.where(keep, g, 0.0)
        prior_diag = None
        if has_priors:
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
            prior_diag = jnp.where(keep, prior_inv2, 0.0)
            gm = gm + jnp.where(
                keep, prior_inv2 * (p_flat - isl["prior_mean"]), 0.0)

        def chi2_at(lam_j):
            if has_priors:
                H_d = (Hm + lam_j * jnp.diag(jnp.diag(Hm))
                       + jnp.diag(prior_diag)
                       + 1e-8 * jnp.eye(nP, dtype=Hm.dtype))
            else:
                H_d = (Hm + lam_j * jnp.diag(jnp.diag(Hm))
                       + 1e-8 * jnp.eye(nP, dtype=Hm.dtype))
            delta = -jnp.linalg.solve(
                H_d.astype(jnp.float64), gm.astype(jnp.float64)
            ).astype(p_flat.dtype)
            delta = jnp.where(keep, delta, 0.0)
            delta = jnp.clip(delta, -jnp.asarray(clip_vec),
                             jnp.asarray(clip_vec))
            p_try = jnp.clip(p_flat + delta, isl["lower"], isl["upper"])
            return chi2_of(p_try, isl, psf)

        best = chi2_at(jnp.float32(1e-9)) - chi2_cur
        for e in range(-8, 7):
            best = jnp.minimum(
                best, chi2_at(jnp.float32(10.0 ** e)) - chi2_cur)
        return best

    def final_fluxes(p_flat, isl, psf):
        """Final joint fluxes + per-member marginalized errors
        (band-at-a-time scan; outputs stacked to (n_bands, N))."""
        p = p_flat.reshape(N, 6)

        def body(c, xs):
            d_b, w_b, psf_b = xs
            m_b = models_band(p, isl, psf_b)
            f, A64 = _fluxes_band(m_b, d_b, w_b, isl["mmask"])
            Ainv = jnp.linalg.inv(A64)             # (N, N)
            ferr = jnp.sqrt(jnp.maximum(
                jnp.diag(Ainv), 1e-30)).astype(jnp.float32)
            return c, (f, ferr)

        _, (f, ferr) = jax.lax.scan(body, 0, _band_xs(isl, psf))
        return f, ferr                             # (n_bands, N) each

    def final_errors(p_flat, isl, psf):
        """Post-convergence nonlinear-parameter covariance (task #28a) +
        the prior_frac disclosure diagonal (R1).

        Rebuilds the UN-DAMPED flux-marginalized GN Hessian H at the
        converged ``p_flat`` through the exact ``hg_scan`` the LM step
        uses (identical traced ops; the gradient is discarded), masks
        it with the STATIC keep -- ``isl['keep'] & mmask``, NOT the
        ``un_freeze_steps``-annealed variant: annealing is a step-time
        optimization and the Sersic index was genuinely fit -- and
        inverts in float64.

        POSTERIOR errors (R1): with priors, the SAME keep-masked prior
        diagonal ``one_step`` adds is added to the masked Hessian
        ``Hm`` BEFORE inversion (mean-independent -- only the curvature
        matters for the covariance), so the reported pos/shape/n errors
        are the POSTERIOR (data + prior) 1-sigma widths, consistent with
        the posterior-MEAN values the solver reports. The prior diagonal
        is added OUTSIDE ``hg_scan`` -- exactly as in ``one_step`` -- so
        the #28a shared-``hg_scan`` trace invariant with the LM step is
        preserved (``hg_scan`` itself is untouched). With no priors this
        block does not run and the returned variances are byte-identical
        to the pre-R1 kernel.

        The diagonal of ``inv(Hm)`` is the MARGINAL variance of each
        kept nonlinear parameter: marginal over the island's other kept
        nonlinear params AND over the per-band fluxes (the VarPro solve
        lives inside ``residual_band``, so AD already applied the
        Golub-Pereyra chain rule); the background stays FROZEN (design
        D1). Frozen/padded slots carry a unit diagonal, so their
        variance reads exactly 1.0 here; the HOST masks them to NaN
        (:func:`island_plan.unpack_class_results`).

        Returns (N, 13) float32: ``[var_x, var_y, var_logre, var_ee1,
        var_ee2, var_un, cov_ee1_ee2, pf_x, pf_y, pf_logre, pf_ee1,
        pf_ee2, pf_un]`` -- the first 7 are the covariance (var
        diagonal + the ee1-ee2 off-diagonal for ELL/THETA propagation);
        the trailing 6 are the per-internal-param prior fractions
        ``prior_inv2_k / (H_kk + prior_inv2_k)`` (data GN diagonal H_kk;
        0 where the param carries no prior -- the HOST maps those to
        NaN). All-NaN in the trailing 6 when the class has no priors.
        """
        H, _ = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)  # (nP,)
        keep_o = keep[:, None] & keep[None, :]
        Hm = jnp.where(keep_o, H, 0.0) + jnp.diag(jnp.where(keep, 0.0, 1.0))
        if has_priors:
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]  # (nP,)
            prior_diag = jnp.where(keep, prior_inv2, 0.0)
            Hm = Hm + jnp.diag(prior_diag)         # POSTERIOR curvature
            # prior_frac_k = prior_inv2_k / (H_kk + prior_inv2_k) on the
            # DATA GN diagonal (kept slots); 0 where no prior/no info.
            hdiag = jnp.where(keep, jnp.diagonal(H), 0.0)
            den = hdiag + prior_diag
            pf6 = jnp.where(den > 0, prior_diag / den, 0.0).reshape(
                N, 6).astype(jnp.float64)
        else:
            pf6 = jnp.full((N, 6), jnp.nan, dtype=jnp.float64)
        Sigma = jnp.linalg.inv(Hm.astype(jnp.float64))
        var6 = jnp.diagonal(Sigma).reshape(N, 6)
        s6 = 6 * jnp.arange(N)
        cov_ee = Sigma[s6 + 3, s6 + 4]             # (N,) ee1-ee2 cross term
        return jnp.concatenate(
            [var6, cov_ee[:, None], pf6], axis=1).astype(jnp.float32)

    def final_flux_marg(p_flat, isl, psf):
        """Root-fix D2: per-band SHAPE-marginalized flux-error INCREMENT.

        The reported ``FLUXERR`` is the conditional flux 1-sigma of the
        VarPro amplitude solve -- marginal over the (linear) flux block but
        CONDITIONAL on the freely-fitted extended-source shape/position.
        For low-S/N extended sources the flux-shape covariance is the
        dominant missing error term (D-diagnosis: UDS S/N<=30 pull_tot
        2.46 -> 1.07 after decorrelating against dlogRe/dell/dx/dy). This
        kernel returns the Schur-complement marginalization increment

            Var_marg(F) = Var_cond(F) + (dF*/dtheta) . Cov(theta) . (dF*/dtheta)^T

        per band per member, where ``dF*/dtheta`` is the exact VarPro
        amplitude sensitivity (AD Jacobian of the closed-form flux solve at
        convergence) and ``Cov(theta) = inv(Hm)`` is the SAME posterior
        (data + prior) flux-marginalized nonlinear covariance ``final_errors``
        inverts -- so no second normal-matrix solve is introduced. The
        block-inverse identity ``[H^-1]_ff = A^-1 + A^-1 B S^-1 B^T A^-1``
        with ``A^-1 B = -dF*/dtheta`` and ``S = inv(Cov)`` makes this the
        exact ff-block of the full joint (flux, theta) inverse.

        MARGINALIZATION SET (zero tuned constants): only EXTENDED members'
        freely-fitted nonlinear parameters (position + shape + free-n). A
        point source has no free shape block, so its columns -- INCLUDING
        its free position -- are excluded, and the per-source ROW is masked
        by ``is_ext`` as well; hence a STAR row's increment is exactly 0 and
        ``FLUXERR_MARG == FLUXERR`` bit-for-bit (even when the star shares an
        island with a galaxy). Returns ``dvar (n_bands, N)`` float32, 0 for
        star rows and padded slots; the HOST forms
        ``FLUXERR_MARG = sqrt(FLUXERR^2 + dvar)``.
        """
        # Cov(theta): reuse final_errors' EXACT posterior covariance Sigma.
        H, _ = hg_scan(p_flat, isl, psf)
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)  # (nP,)
        keep_o = keep[:, None] & keep[None, :]
        Hm = jnp.where(keep_o, H, 0.0) + jnp.diag(jnp.where(keep, 0.0, 1.0))
        if has_priors:
            prior_inv2 = isl["prior_inv_sigma"] * isl["prior_inv_sigma"]
            Hm = Hm + jnp.diag(jnp.where(keep, prior_inv2, 0.0))
        Sigma = jnp.linalg.inv(Hm.astype(jnp.float64))            # (nP, nP)
        # Column mask: EXTENDED members' free params only. keep already gates
        # shape (2:5) to is_ext and n (5) to fn_flag; ANDing is_ext also drops
        # a star member's free position (0:2). A star member -> all 6 cols 0.
        is_ext = (isl["stype"] != SOURCE_STAR) & isl["mmask"]     # (N,)
        marg_keep = (keep.reshape(N, 6) & is_ext[:, None]).reshape(-1)
        mk = marg_keep.astype(jnp.float64)                       # (nP,)

        def dvar_band(carry, xs):
            d_b, w_b, psf_b = xs
            def fstar(q):
                m_b = models_band(q.reshape(N, 6), isl, psf_b)
                f_b, _ = _fluxes_band(m_b, d_b, w_b, isl["mmask"])
                return f_b                                       # (N,)
            Jb = jax.jacrev(fstar)(p_flat).astype(jnp.float64)   # (N, nP)
            Jm = Jb * mk[None, :]                                # mask columns
            # rowwise quadratic form diag(Jm Sigma Jm^T)
            dv = jnp.einsum("ka,ab,kb->k", Jm, Sigma, Jm)        # (N,)
            # ROW mask: star rows -> 0 (a star blended with a galaxy would
            # otherwise pick up the galaxy's shape via dF_star/dtheta_gal).
            dv = jnp.where(is_ext, jnp.maximum(dv, 0.0), 0.0)
            return carry, dv.astype(jnp.float32)

        _, dvar = jax.lax.scan(dvar_band, 0, _band_xs(isl, psf))
        return dvar                                              # (n_bands, N)

    def residual_coadd(p_flat, isl, psf):
        """Task #34: inverse-variance-weighted band-COADDED model-subtracted
        residual of ONE island, on its (B,B) bbox window. Reuses the exact
        ``models_band`` + VarPro ``_fluxes_band`` the LM/chi2 path uses, so
        the subtracted model is the converged fit. Returns ``(Rnum, W)``:
        ``Rnum = sum_b w_b*(d_b - model_b)`` and ``W = sum_b w_b``. The host
        forms the coadd residual ``R = Rnum / W`` with per-pixel noise
        ``sigma = 1/sqrt(W)`` (masked pixels W=0 -> excluded from SEP), the
        NOISE-SCALED detection surface for un-attributed light."""
        def body(carry, xs):
            Rn, Wc = carry
            d_b, w_b, psf_b = xs
            p = p_flat.reshape(N, 6)
            m_b = models_band(p, isl, psf_b)
            f_b, _ = _fluxes_band(m_b, d_b, w_b, isl["mmask"])
            model_img = jnp.einsum("k,kij->ij", f_b, m_b)
            return (Rn + w_b * (d_b - model_img), Wc + w_b), None
        (Rn, Wc), _ = jax.lax.scan(
            body,
            (jnp.zeros((B, B), dtype=jnp.float32),
             jnp.zeros((B, B), dtype=jnp.float32)),
            _band_xs(isl, psf))
        return Rn, Wc

    def source_metrics(p_flat, isl, psf):
        """Task #17: per-SOURCE model-quality metrics at the converged fit
        (legacypipe DR9 fracflux/fracin/fracmasked/rchisq convention, computed
        on the island (B,B) bbox). Read-only -- reuses the exact ``models_band``
        + VarPro ``_fluxes_band`` + ``model_img`` the LM/chi2 path uses, so it
        scores the converged fit; no fitting, no extra render pass beyond this
        band scan. Returns ``(N, 4)`` float32: ``[rchisq, fracflux, fracin,
        fracmasked]`` per member.

        - ``rchisq_k``: own-profile-weighted reduced chi2 of the FULL-model
          whitened residual, per-SOURCE. Unlike the per-ISLAND ``chi2_nu`` (which
          is blind to member-to-member misattribution because the joint model
          still fits the pixels), this restricts the chi2 to source k's own model
          footprint -> the fair survey fit-quality axis (DESI dchisq / HSC CHI2 /
          SExtractor CHI2) we were missing. ~1 for a clean fit, >1 for a model
          misfit near k.
        - ``fracflux_k``: profile-weighted fraction of source k's aperture flux
          contributed by OTHER island members (blend contamination). ~0 isolated.
        - ``fracin_k``: fraction of source k's analytic-total model flux contained
          in the fitted (B,B) footprint (<1 when large-Re wings exceed the bbox).
        - ``fracmasked_k``: profile-weighted fraction of source k's model flux
          landing on masked (invvar==0) pixels.

        Multi-band coadd matches legacypipe: fracflux/fracmasked use a
        fin-weighted mean (den = sum_b fin_b, fin_b = clip(sum(unit model),0,1)),
        fracin/rchisq use sum_b(num)/sum_b(den). Padded slots -> masked to
        0 (fracflux/fracmasked) / NaN (fracin/rchisq) by the host.

        #35 Option A: additionally returns ``fwin (n_bands, N)`` -- the
        per-band DATA-CONSTRAINED window flux ``f_kb * sum(m_kb)`` (the
        island's joint VarPro amplitude times the unit-model window sum,
        i.e. the per-band quantity behind fracin). The reported
        FLUX_{band}_fit is the untruncated analytic-total extrapolation;
        fwin is the part the window pixels actually constrained. Same
        read-only scan (no fitting); padded slots -> NaN."""
        p = p_flat.reshape(N, 6)
        z = jnp.zeros((N,), dtype=jnp.float32)

        def body(carry, xs):
            (ff_n, ff_d, fin_n, fin_d, fm_n, fm_d, rc_n, rc_d) = carry
            d_b, w_b, psf_b = xs
            m_b = models_band(p, isl, psf_b)                 # (N,B,B) unit models
            f_b, _ = _fluxes_band(m_b, d_b, w_b, isl["mmask"])   # (N,)
            model_img = jnp.einsum("k,kij->ij", f_b, m_b)    # (B,B) total model
            own = f_b[:, None, None] * m_b                   # (N,B,B) own (scaled)
            s_m = jnp.sum(m_b, axis=(1, 2))                  # (N,) contained frac
            fin = jnp.clip(s_m, 0.0, 1.0)                    # multi-image weight
            # fracflux = Sum((total-own)*own)/Sum(own^2), fin-weighted
            neigh = model_img[None] - own                    # (N,B,B) neighbour light
            ff_bot = jnp.sum(own * own, axis=(1, 2))         # (N,)
            ff_ratio = jnp.where(ff_bot > 0,
                                 jnp.sum(neigh * own, axis=(1, 2)) / ff_bot, 0.0)
            # fracmasked = Sum((w==0)*unit model) clipped, fin-weighted
            masked = (w_b == 0.0).astype(jnp.float32)        # (B,B)
            fm_b = jnp.clip(jnp.sum(masked[None] * m_b, axis=(1, 2)), 0.0, 1.0)
            # rchisq = own-profile-weighted per-pixel chi2 (w*resid^2)
            chi_pix = w_b * (d_b - model_img) ** 2           # (B,B)
            rc_top = jnp.sum(m_b * chi_pix[None], axis=(1, 2))   # (N,)
            return ((ff_n + fin * ff_ratio, ff_d + fin,
                     fin_n + jnp.abs(s_m), fin_d + jnp.ones_like(s_m),
                     fm_n + fin * fm_b, fm_d + fin,
                     rc_n + jnp.abs(rc_top), rc_d + jnp.abs(s_m)),
                    f_b * s_m)      # #35: per-band window flux (ys)

        ((ff_n, ff_d, fin_n, fin_d, fm_n, fm_d, rc_n, rc_d),
         fwin_b) = jax.lax.scan(
            body, (z, z, z, z, z, z, z, z), _band_xs(isl, psf))
        mm = isl["mmask"]
        nan = jnp.float32(jnp.nan)
        fracflux = jnp.where((ff_d > 0) & mm, ff_n / ff_d, 0.0)
        fracin = jnp.where((fin_d > 0) & mm, fin_n / fin_d, nan)
        fracmasked = jnp.where((fm_d > 0) & mm, fm_n / fm_d, 0.0)
        rchisq = jnp.where((rc_d > 0) & mm, rc_n / rc_d, nan)
        fwin = jnp.where(mm[None, :], fwin_b, nan)            # (n_bands, N)
        return (jnp.stack([rchisq, fracflux, fracin, fracmasked],
                          axis=1).astype(jnp.float32),        # (N, 4)
                fwin.astype(jnp.float32))

    def railed_probe(p_flat, isl):
        """Bound-pinned disclosure for step_mode="trust_region": True when the
        converged fit pinned ANY kept free param at its lower/upper box bound
        (Re ceiling / Sersic-n cap / position / ellipticity bound). The dogleg
        removes the fixed-clip-saturation signature diverged_probe keys on
        (rejects now change the proposal), so this is its trust-region analogue
        -- read-only, post-hoc, parameter-free (exact bound equality after the
        box clip). Disclosure only (island_railed): never gates
        fit_is_reliable."""
        keep = (isl["keep"] & isl["mmask"][:, None]).reshape(-1)
        at_bound = keep & ((p_flat <= isl["lower"]) | (p_flat >= isl["upper"]))
        return jnp.any(at_bound)

    if convergence == "dlnp":
        # dlnp step (11-state changed-mode layout, f64 chi2 carry).
        # Validation above already excluded trust_region and "changed".
        step_v = jax.jit(jax.vmap(
            one_step_dlnp,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)))
    elif step_mode == "trust_region":
        step_v = jax.jit(jax.vmap(
            one_step_tr, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)))
    elif stall_count_mode == "changed":
        step_v = jax.jit(jax.vmap(
            one_step_changed,
            in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None)))
    else:
        step_v = jax.jit(jax.vmap(
            one_step, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None)))
    # dlnp mode: the LOOP's chi2 (fresh-chunk init + step carry) is the
    # f64-carry flavor; every read-only probe (plateau/diverged/metrics)
    # keeps the f32 chi2_of so the certificate arithmetic is identical
    # across modes.
    chi2_v = jax.jit(jax.vmap(
        chi2_of64 if convergence == "dlnp" else chi2_of,
        in_axes=(0, 0, None)))
    final_v = jax.jit(jax.vmap(final_fluxes, in_axes=(0, 0, None)))
    err_v = jax.jit(jax.vmap(final_errors, in_axes=(0, 0, None)))
    # D2: post-convergence SHAPE-marginalized flux-error increment (opt-in
    # via compute.gpu.error_model.marginal_flux_err_enabled; kernel built
    # unconditionally like err_v/metrics_v but only CALLED when the flag is
    # on, so an off run never compiles/executes it -> byte-identical).
    fluxmarg_v = jax.jit(jax.vmap(final_flux_marg, in_axes=(0, 0, None)))
    residual_v = jax.jit(jax.vmap(residual_coadd, in_axes=(0, 0, None)))
    metrics_v = jax.jit(jax.vmap(source_metrics, in_axes=(0, 0, None)))
    diverged_v = jax.jit(jax.vmap(diverged_probe,
                                  in_axes=(0, 0, 0, 0, None)))
    plateau_v = jax.jit(jax.vmap(plateau_probe, in_axes=(0, 0, 0, None)))
    railed_v = jax.jit(jax.vmap(railed_probe, in_axes=(0, 0)))

    # #19a: post-convergence Re PROFILE-LIKELIHOOD scan (read-only,
    # opt-in via error_model.re_profile_enabled). chi2 along log-Re
    # offsets for every member slot; per-band fluxes are RE-SOLVED at
    # each scan point (the VarPro solve lives inside residual_band), so
    # this traces the flux-profiled Re valley -- the honest Re width
    # along the flux-Re degeneracy direction, which the local-Fisher
    # re_err cannot see when the valley is curved (wanderer Re pull ~9
    # with covered flux, ledger 2026-07-21k). Cost: N*K extra chi2
    # evals per island (star slots wasted -- vmapped cond would compute
    # both branches anyway); enable deliberately.
    _reprof_d = jnp.asarray(REPROF_DELTAS_LN)

    def re_profile(p_flat, isl, psf):
        base = chi2_of(p_flat, isl, psf)

        def one_slot(n):
            idx = n * 6 + 2

            def one_d(dk):
                return chi2_of(p_flat.at[idx].add(dk), isl, psf)
            # lax.map (sequential over K), NOT vmap: vmapping K multiplies
            # the chi2 working set by K on top of the island-chunk vmap
            # (measured 59.7 GiB OOM at K=12 on the color scene).
            return jax.lax.map(one_d, _reprof_d)
        prof = jax.lax.map(one_slot, jnp.arange(N))
        return (prof - base).astype(jnp.float32)     # (N, K)
    reprof_v = jax.jit(jax.vmap(re_profile, in_axes=(0, 0, None)))

    # #35 scene-subtract: per-island member ADD-BACK kernel. Renders this
    # island's members at their SEED params (p0) and returns
    # Sum_k member_s1_flux[k,b] * models_band(p0)[k] per band -- the exact model
    # the field-wide pass subtracted for these members, so adding it back to the
    # (scene-subtracted) gathered cutout restores member light at TRUE level
    # while NON-member light stays removed. Built ONLY when scene_subtract (else
    # None -> run_class_chunk skips it and the compiled fit program is byte-
    # identical to the pre-scene-subtract kernel). Mirrors chi2_of's per-band
    # lax.scan + residual_band's einsum("k,kij->ij", f, m) model coadd.
    addback_v = None
    if scene_subtract:
        def addback_of(p_flat, isl, psf):
            p = p_flat.reshape(N, 6)
            msf_bn = isl["member_s1_flux"].T           # (n_bands, N)

            def _add_band(carry, xs):
                psf_b, msf_b = xs
                m_b = models_band(p, isl, psf_b)       # (N, B, B)
                return carry, jnp.einsum("k,kij->ij", msf_b, m_b)

            _, adds = jax.lax.scan(_add_band, jnp.float32(0.0), (psf, msf_bn))
            return adds                                # (n_bands, B, B)

        addback_v = jax.jit(jax.vmap(addback_of, in_axes=(0, 0, None)))

    def _gather_one(stack_dev, oy_i, ox_i):
        return jax.lax.dynamic_slice(
            stack_dev, (jnp.int32(0), oy_i, ox_i), (n_bands, B, B))

    _gather_jit = jax.jit(jax.vmap(_gather_one, in_axes=(None, 0, 0)))

    def gather(stack_dev, oy, ox):
        """(n_bands, Hp, Wp) padded device stack -> (n_isl, n_bands, B, B).
        Jitted ONCE per kernel-cache key (rule 1)."""
        return _gather_jit(stack_dev,
                           jnp.asarray(oy, dtype=jnp.int32),
                           jnp.asarray(ox, dtype=jnp.int32))

    # #35: diverged_v is the post-hoc probe (legacy mode); the trailing
    # string records the static STEP PROTOCOL so run_class_chunk selects
    # the matching loop (legacy "proposals" 8-state, byte-identical
    # program + post-hoc probe; "changed" 11-state in-kernel tracker;
    # "dlnp" 11-state in-kernel tracker with the f64 chi2 carry).
    step_protocol = "dlnp" if convergence == "dlnp" else str(stall_count_mode)
    # slot 15 (dlnp only): the statistical quantum 2*plateau_dlnp_tol --
    # run_class_chunk applies it to the ladder certificate to map
    # reject-exits to converged-on-plateau vs honest-stall (E1a).
    _dlnp_quantum = (float(2.0 * plateau_dlnp_tol)
                     if convergence == "dlnp" else None)
    out = (gather, step_v, chi2_v, final_v, err_v, residual_v, metrics_v,
           diverged_v, step_protocol, fluxmarg_v, railed_v,
           str(step_mode), addback_v, reprof_v, plateau_v, _dlnp_quantum)
    _KERNEL_CACHE[key] = out
    return out


def build_psf_tree(psfs: list, tier: int) -> tuple[dict, int]:
    """Stack the per-band PSF polynomials (zero-padded to ``tier``) into
    the traced-pytree layout the bucket kernels expect. Returns
    ``(psf_tree, n_terms)``. Device upload happens at first kernel call."""
    import jax.numpy as jnp

    padded = [_pad_psf_np(p, int(tier)) for p in psfs]
    n_terms = int(padded[0].coefs.shape[-1])
    psf_tree = {
        "coefs": jnp.asarray(np.stack(
            [p.coefs for p in padded], axis=0), dtype=jnp.float32),
        "x_origin": jnp.asarray(
            [p.x_origin for p in padded], dtype=jnp.float32),
        "x_scale": jnp.asarray(
            [p.x_scale for p in padded], dtype=jnp.float32),
        "y_origin": jnp.asarray(
            [p.y_origin for p in padded], dtype=jnp.float32),
        "y_scale": jnp.asarray(
            [p.y_scale for p in padded], dtype=jnp.float32),
    }
    return psf_tree, n_terms


def make_class_kernels(
    plan, n_bands: int, n_terms: int, *,
    render_backend: str, reg: float, pos_step_max_pix: float,
    chi2_ftol_rel: float, step_tol_pix: float, un_freeze_steps: int,
    star_render_split: bool = True,
    has_priors: bool = False,
    stall_count_mode: str = "proposals",
    step_mode: str = "clip",
    scene_subtract: bool = False,
    crossband: dict | None = None,
    convergence: str = "legacy",
    plateau_dlnp_tol: float = 0.5,
):
    """Bucket kernels for one :class:`~.island_plan.IslandClassPlan`
    (thin wrapper over :func:`_make_bucket_kernels` with the production
    step/tolerance constants). ``has_priors`` (from
    ``class_host_arrays``) selects the prior-injected vs the byte-
    identical pre-prior program for this class. ``star_render_split``
    (PERF #14, ``compute.gpu.solver.island.star_render_split``) toggles
    the mixed-class star/galaxy render split; ``False`` emits the OLD
    single-N-wide render (byte-identical to pre-#14)."""
    return _make_bucket_kernels(
        B=int(plan.bbox_class), T=int(plan.tier), N=int(plan.nsrc_class),
        n_bands=int(n_bands), n_terms=int(n_terms),
        tangent_chunk=int(plan.tangent_chunk),
        render_kind=str(plan.kind), n_gal=int(plan.n_gal_class),
        star_render_split=bool(star_render_split),
        render_backend=str(render_backend),
        reg=float(reg), pos_step_max=float(pos_step_max_pix),
        shape_step_max=DEFAULT_SHAPE_STEP_MAX,
        un_step_max=DEFAULT_UN_STEP_MAX,
        ftol_rel=float(chi2_ftol_rel), step_tol=float(step_tol_pix),
        shape_step_tol=DEFAULT_SHAPE_STEP_TOL,
        un_freeze_steps=int(un_freeze_steps),
        has_priors=bool(has_priors),
        stall_count_mode=str(stall_count_mode),
        # STAR-only classes stay "clip" (bit-identical) even with the flag on;
        # the trust-region step only applies to galaxy-bearing classes.
        step_mode=(str(step_mode) if str(plan.kind) != "star" else "clip"),
        # scene-subtract applies to ALL classes (every island's members must be
        # added back, star-only included) -- NOT star-exempted like step_mode.
        scene_subtract=bool(scene_subtract),
        crossband=crossband,
        # dlnp convergence applies to ALL classes incl. star-only (the
        # measured prong-2 leaked-descent channel IS bright stars) --
        # NOT star-exempted like step_mode.
        convergence=str(convergence),
        plateau_dlnp_tol=float(plateau_dlnp_tol),
    )


_KERNEL_FINGERPRINT: "str | None" = None


def kernel_fingerprint() -> str:
    """Content hash of the sources that DEFINE the compiled island/grid kernels.

    effarc2 R3 (2026-08-28). The island width cache and the cross-tile XLA cost
    table both identify their entries with ``engine`` =
    the package version (``tract7dt.__version__``). That string is the release
    version, and this repository is developed as an uncommitted
    working tree that is shipped by copying files -- so the version does not
    move when the kernels do. Measured across exactly such a change (the
    effarc2 ship): identical ``engine``/``jax``/device metadata and identical
    class keys, while the same class at the same rung reported 2.4x-5.4x less
    device memory. A cache written before the change is therefore adopted
    after it, and (with the cost table feeding ``_probe``'s rung-1 answer) a
    stale number can decide a SOLVER PATH, not just a width.

    Hashing the kernel sources makes both caches self-invalidating on any edit
    to the code that determines the compiled programs -- which is what the
    ``engine`` field was always meant to express. Cheap (a few hundred KB,
    once per process) and derived, not tuned. If a file cannot be read the
    fingerprint degrades to ``"unknown"``, which is a distinct key of its own
    rather than a silent match.
    """
    global _KERNEL_FINGERPRINT
    if _KERNEL_FINGERPRINT is not None:
        return _KERNEL_FINGERPRINT
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    parts = [
        os.path.join(here, "island_solver.py"),
        os.path.join(here, "island_plan.py"),
        os.path.join(here, "source_model.py"),
        os.path.join(here, "psf_polynomial.py"),
        os.path.join(here, "sersic_mog.py"),
        os.path.join(here, "forced_phot_sparse.py"),
        os.path.abspath(os.path.join(
            here, os.pardir, "optimize", "gpu", "kernels", "lanczos.py")),
    ]
    h = hashlib.blake2b(digest_size=8)
    try:
        for fp in parts:
            with open(fp, "rb") as fh:
                h.update(fh.read())
        _KERNEL_FINGERPRINT = h.hexdigest()
    except Exception:                                          # noqa: BLE001
        _KERNEL_FINGERPRINT = "unknown"
    return _KERNEL_FINGERPRINT


_MEMPROBE = os.environ.get("TRACT7DT_MEMPROBE", "") not in ("", "0")


def _memprobe(tag: str, fn, args: tuple, width: int, ident: str = "") -> None:
    """T5/#28 diagnostic (env ``TRACT7DT_MEMPROBE``; OFF by default and
    RESULT-NEUTRAL when on): log what XLA itself says this compiled kernel
    needs, at the chunk width actually in use.

    ``ident`` is the BUCKET identity ``B=.. N=.. kind=.. t=..``: the width
    alone is uninterpretable, because the term that killed T08285
    (``final_flux_marg``'s ``width x N^2 x B^2`` jacrev intermediate) scales
    with the bucket, not with the chunk.

    Kept after the T5/#28 fix as the field-run instrument: the chunk width is
    now chosen by :func:`measure_class_chunk_width`, and this is how you see,
    from a real run's log, what each kernel actually asked for at the width it
    got. It is also the record of how the hand-calibrated byte model was caught
    (it had no fixed-cost term AND was calibrated through ``island_fit``, the
    NON-sharded entry point, which cannot even run ``fluxmarg_v`` -- the kernel
    that asked for 19.68 GiB and killed the crowded tile).

    ``lower(...).compile()`` hits the persistent JIT cache for an
    already-compiled signature, so this is close to free; any failure is
    swallowed -- a diagnostic must never be able to break a run.
    """
    if not _MEMPROBE:
        return
    try:
        t0 = time.perf_counter()
        comp = fn.lower(*args).compile()
        dt = time.perf_counter() - t0
        m = comp.memory_analysis()
        # print, not logging: this module has no logger, and the probe runs
        # inside SPAWNED shard workers whose stdout is captured to run.log.
        # compile_s is ~0 on a persistent-JIT-cache hit and the COLD compile
        # cost otherwise (= what a ladder rung costs).
        print(f"memprobe {tag}: {ident} width={width} "
              f"temp={m.temp_size_in_bytes / 2 ** 20:.1f} MiB "
              f"arg={m.argument_size_in_bytes / 2 ** 20:.1f} "
              f"out={m.output_size_in_bytes / 2 ** 20:.1f} "
              f"alias={m.alias_size_in_bytes / 2 ** 20:.1f} "
              f"compile_s={dt:.2f}", flush=True)
    except Exception as exc:                                   # noqa: BLE001
        print(f"memprobe {tag}: {ident} width={width} unavailable "
              f"({type(exc).__name__}: {str(exc)[:100]})", flush=True)


# ---------------------------------------------------------------------------
# T5/#28 measured chunk width -- ASK XLA, don't model the bytes.
#
# What this replaces: island_plan._island_step_bytes, eight hand-fitted
# coefficients times a hand-set GB budget. Three measured defects
# (2026-08-02, development ledger 6-4):
#   (a) the model is `per-island bytes x chunk` with no fixed-cost term,
#       so a small-chunk calibration folds the fixed cost into the slope
#       and over-charges up to 6x at wide chunks (chunks too small = slow);
#   (b) it was calibrated through island_fit, the NON-sharded path, which
#       has no compute_flux_marg argument and therefore never runs the
#       marginal-flux kernel that production DOES run;
#   (c) that kernel's jacrev intermediate is `width x N^2 x B^2` -- N is a
#       BUCKET property and width is the chunk, so their product cannot be
#       folded into any per-island constant. Measured here on a
#       (128, 32, star, t55) class: fluxmarg_v needs 193 MiB/island where
#       step_v needs 61, and 103 x 193 MiB = the 19.68 GiB that killed the
#       crowded tile.
#
# The replacement measures BOTH axes instead of assuming either:
#   budget = free device memory, read from the device;
#   cost   = what XLA reports for EVERY kernel that will actually run at
#            that width, maximum taken.
# The search only compiles rungs it then measures -- no extrapolation is
# ever adopted (a fitted form breaks on data we have not seen, and uniphot
# is not a 7DT-only tool).
# ---------------------------------------------------------------------------

# Ladder = powers of two. Two reasons, neither of them tuned to our data:
# the search is O(log width), and quantised widths RECUR across fields, so
# the persistent JIT cache carries them from one tile to the next instead of
# every tile paying to compile its own arbitrary widths.
#
# The CEILING is the one number here that is not derived. It is not a memory
# constant (memory is measured) -- it bounds SHAPE DIVERSITY to 13 rungs.
# Inherited from the pre-2026-08-02 `min(4096, ...)`.
# 2026-08-14 (width bundle #1): a topology-derived cap (>= n_devices chunks
# per big class, via ``effective_rung_limit``'s min_chunks term) was tried
# and MEASURED NEGATIVE -- width is the vmap batch, so thinner chunks halved
# GPU throughput (sparse fit stage 66.4 -> 142.2 s, total busy ~2x) despite
# better balance. min_chunks stays available but ships as 1 (= the
# historical limit). A future cap must weigh the measured isl/s rate, not
# chunk counts. Still an open item; the value itself remains underived.
_LADDER_MAX_WIDTH = 4096


def effective_rung_limit(n_islands: int, max_width: int = _LADDER_MAX_WIDTH,
                         min_chunks: int = 1) -> int:
    """Largest ladder rung a class may adopt.

    ``min_chunks`` (width bundle #1) is the scheduling-topology term: when
    the class holds at least ``min_chunks`` islands, its width is capped so
    it yields at least that many chunks -- a schedulable unit set must not
    be smaller than the executor set it feeds. Derived from the device
    count; no free constant. Below that population the class cannot occupy
    every executor anyway and the cap does not apply. ``_LADDER_MAX_WIDTH``
    stays as the shape-diversity hard cap.
    """
    limit = max(1, min(int(max_width), max(1, int(n_islands))))
    if int(min_chunks) > 1 and int(n_islands) >= int(min_chunks):
        limit = min(limit, max(1, int(n_islands) // int(min_chunks)))
    return 1 << (int(limit).bit_length() - 1)

# Measurement compiles skip autotuning: measured 2026-08-02 to report
# byte-identical temp/argument/output at 2.5-3x lower compile cost, and
# autotuning is the stage that FAILED ("No valid config found") on the
# over-large width -- a probe must not be able to fail where the thing it
# is protecting against fails.
_NO_AUTOTUNE = {"xla_gpu_autotune_level": "0"}


def _device_free_bytes_nvml() -> int:
    """FREE memory on this process's PHYSICAL card, from NVML only. 0 if unknown.

    Deliberately does NOT consult JAX's allocator stats. Those describe an
    arena, not the card: under the BFC allocator ``bytes_limit - bytes_in_use``
    is what THIS process may still take from its own reservation and says
    nothing about a co-resident job, and the field-stack pre-flight in
    ``island_stage._shard_setup`` needs the card-wide number or it is either
    dead code or -- worse -- a false alarm that kills a healthy run.
    ``_device_free_bytes`` keeps the arena-first behaviour for the chunk-width
    ladder, which is sizing an allocation inside that same arena.

    ``nvidia-smi`` ignores ``CUDA_VISIBLE_DEVICES``, so ``-i <CVD>`` selects
    the physical card that ``_worker_init`` pinned this worker to.
    """
    try:
        import subprocess
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        cmd = ["nvidia-smi", "--query-gpu=memory.free",
               "--format=csv,noheader,nounits"]
        if vis:
            cmd += ["-i", vis]
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10, check=True).stdout
        return int(float(out.strip().splitlines()[0])) * 2 ** 20
    except Exception:                                          # noqa: BLE001
        return 0


def _device_free_bytes() -> int:
    """FREE memory on THIS process's GPU, measured. 0 if unknown.

    Two sources so the answer does not depend on which allocator the
    worker was started with: JAX's own stats (BFC arena) first, then NVML
    via :func:`_device_free_bytes_nvml` (the platform allocator returns no
    stats -- checked 2026-08-02 -- and that is the allocator the island
    shard workers use).
    """
    try:
        import jax
        ms = jax.local_devices()[0].memory_stats()
        if ms:
            lim = int(ms.get("bytes_limit", 0))
            use = int(ms.get("bytes_in_use", 0))
            if lim > 0:
                return max(0, lim - use)
    except Exception:                                          # noqa: BLE001
        pass
    return _device_free_bytes_nvml()


def _gpu_occupants(index: "str | int | None" = None) -> list[tuple[int, int, str]]:
    """``[(pid, used_bytes, username), ...]`` for the CUDA compute processes
    on ONE PHYSICAL GPU. Attribution only -- never control flow.

    Companion to :func:`_device_free_bytes` above: that one answers "how much
    is left", this one answers "who took it". Without it a contended card
    produces ``RESOURCE_EXHAUSTED ... on device ordinal 0`` and nothing else,
    because every shard/grid worker runs under ``CUDA_VISIBLE_DEVICES=<its
    card>`` and therefore sees exactly one device, whose ordinal is always 0
    (measured: the 2026-08-29 02:14 and 2026-08-30 03:05 crowded-gate deaths,
    whose logs name neither the card nor the occupant).

    ``index`` is the PHYSICAL index. ``nvidia-smi`` IGNORES
    ``CUDA_VISIBLE_DEVICES`` (verified 2026-08-30 on the development server:
    ``CUDA_VISIBLE_DEVICES=2 nvidia-smi --query-gpu=index`` still prints
    0/1/2) -- the same property ``_device_free_bytes`` already relies on when
    it hands its own CVD value to ``-i``. ``_worker_init`` rewrites CVD to the
    physical index, so the default is right inside a worker.

    The username is the owner of ``/proc/<pid>`` (world-readable), so no extra
    subprocess is spawned per pid. Empty list on any failure: this only ever
    decorates a message.
    """
    import pwd
    import subprocess
    try:
        if index is None:
            index = os.environ.get(
                "CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        cmd = ["nvidia-smi", "--query-compute-apps=pid,used_memory",
               "--format=csv,noheader,nounits"]
        if str(index) not in ("", "?"):
            cmd += ["-i", str(index)]
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10, check=True).stdout
    except Exception:                                          # noqa: BLE001
        return []
    rows: list[tuple[int, int, str]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        try:
            pid, used = int(parts[0]), int(float(parts[1])) * 2 ** 20
        except ValueError:
            continue
        try:
            user = pwd.getpwuid(os.stat(f"/proc/{pid}").st_uid).pw_name
        except Exception:                                      # noqa: BLE001
            user = "?"
        rows.append((pid, used, user))
    return rows


def _xla_call_bytes(fn, args) -> int | None:
    """Device bytes ONE call of this jitted kernel needs, per XLA itself.

    ``temp + argument + output``: the executable's scratch plus the buffers
    that must be live around it. VALIDATED 2026-08-02 against the card:
    over a 64x width ladder on a (128, 32, star, t55) class this predicted
    the steady-state NVML delta to within 0.6% at every rung (200/404/802/
    1598/3194/6386/12804 MiB measured vs 199/400/797/1590/3177/6351/12731
    predicted). It is the right meter -- it just does not cover the
    AUTOTUNER's transient (a further ~+33%, one-off per compiled shape),
    which is why the search ends with a real autotuned compile.
    None when XLA cannot compile or will not report (caller treats that
    exactly like "does not fit").
    """
    try:
        comp = fn.lower(*args).compile(_NO_AUTOTUNE)
        m = comp.memory_analysis()
    except Exception:                                          # noqa: BLE001
        return None
    if m is None:
        return None
    return int(m.temp_size_in_bytes + m.argument_size_in_bytes
               + m.output_size_in_bytes)


def ladder_walk(probe, limit: int, budget_bytes: int, c1: int, dom1: str):
    """The T5/#28 rung search as a PURE function of a probe oracle.

    ``probe(w) -> (cost_bytes | None, dominant_tag)``. Jump to the rung the
    measured w=1 rate allows (never above ``limit``), walk down until a rung
    fits, then grow while the rate at the accepted rung says there is room
    -- every candidate is measured before it is accepted, nothing is
    extrapolated. Returns ``(best_w, best_c, best_dom)``. Factored out of
    :func:`measure_class_chunk_width` for T6/I1 (2026-08-23) so the same
    walk can be run at the class ceiling to fill the cross-tile cost table;
    the body is the 2026-08-02 loop verbatim (test_island_chunk_width pins
    it against the inline original).
    """
    best_w, best_c, best_dom = 1, c1, dom1
    w = min(int(limit),
            1 << max(0, (int(budget_bytes) // max(int(c1), 1)).bit_length() - 1))
    while w > best_w:
        c, dm = probe(w)
        if c is not None and c <= budget_bytes:
            best_w, best_c, best_dom = w, c, dm
            break
        w //= 2
    while best_w * 2 <= int(limit):
        nxt = best_w * 2
        if best_c * 2 > budget_bytes:      # rate at best_w, extended to nxt
            break
        c, dm = probe(nxt)
        if c is None or c > budget_bytes:
            break
        best_w, best_c, best_dom = nxt, c, dm
    return best_w, best_c, best_dom


def measure_class_chunk_width(
    kernels: tuple,
    psf_tree: dict,
    host: dict,
    n_islands: int,
    dstack_dev,
    wstack_dev,
    pad: int,
    *,
    budget_bytes: int = 0,
    compute_param_errors: bool = False,
    compute_source_metrics: bool = False,
    compute_flux_marg: bool = False,
    compute_re_profile: bool = False,
    want_residual: bool = False,
    max_width: int = _LADDER_MAX_WIDTH,
    min_chunks: int = 1,
    cost_table: dict | None = None,
    class_ladder: bool = False,
    verify_compile: bool = True,
    log_fn=None,
) -> dict:
    """Largest ladder width whose MEASURED demand fits the MEASURED budget.

    Returns ``{"width", "cost_bytes", "budget_bytes", "per_island_bytes",
    "probes", "wall_s", "dominant", "verified"}``. ``width == 0`` means the
    class does not fit even one island at a time -- memory-infeasible for
    the joint LM, so the caller routes it to the giant path (seed geometry,
    no island fit; the measured replacement for the byte model's giant
    downgrade).

    ``class_ladder`` (T6/I1): after the tile-limited search, walk the same
    ladder to the ceiling through the memoised probe so the cross-tile cost
    table learns the class's rungs; the returned width is unchanged.

    Two phases, both measured:

    1. CHEAP (autotune off, ~10 s/rung, byte-identical memory report):
       probe rung 1 for the per-island rate, jump to the rung that rate
       allows, then walk -- down if it does not fit, up while the rate at
       the ACCEPTED rung says there is room. The rate only ever chooses
       which rung to MEASURE; no unmeasured width is adopted.
    2. REAL (``verify_compile``): compile the production kernels at the
       chosen width exactly as production will -- autotuner on, chunk
       arguments resident. The autotuner's own scratch is ~+33% on top of
       the executable's demand and lives outside XLA's report, so this is
       the only honest test of "can this machine actually build it". A
       memory-class failure steps down a rung and retries. Nothing is
       wasted: the surviving compiles ARE the production compiles, and
       they land in the persistent JIT cache for every later field.
    """
    import jax.numpy as jnp

    t0 = time.perf_counter()
    (gather, step_v, chi2_v, final_v, err_v, residual_v,
     metrics_v) = kernels[:7]
    kmode = kernels[8] if len(kernels) > 8 else "proposals"
    fluxmarg_v = kernels[9] if len(kernels) > 9 else None
    addback_v = kernels[12] if len(kernels) > 12 else None
    reprof_vk = kernels[13] if len(kernels) > 13 else None
    N = int(host["mmask"].shape[1])
    nP = 6 * N
    n_islands = int(n_islands)
    if budget_bytes <= 0:
        budget_bytes = _device_free_bytes()

    limit = effective_rung_limit(n_islands, max_width, min_chunks)

    # effarc2 K2b (2026-08-26): a class with NO extended member never calls
    # fluxmarg_v (run_class_chunk short-circuits it to the all-zero array the
    # kernel would return by construction), so it must not be PROBED or
    # VERIFY-compiled either. That kernel is both the memory-dominant one
    # (its jacrev intermediate is width x N^2 x B^2 -- the 19.68 GiB that
    # killed T08285 in 2026-08) and the most expensive to compile (measured
    # 14.9 s vs 8.7 s for step_v on the sparse anchor class), so dropping it
    # for star-only classes removes ~3 of the ~18 compiles per class AND lets
    # the ladder walk adopt a wider rung.
    #
    # Cache-key safety: "no extended member" is exactly the class's `kind`
    # ("star" vs "mixed"), which is ALREADY part of the per-class width-cache
    # key, so the recorded verdict stays self-describing and pass B reuses it
    # instead of re-probing. (Getting that wrong would cost a full re-probe:
    # +129 s sparse / +440 s crowded.)
    if compute_flux_marg and fluxmarg_v is not None:
        if not bool(np.any(np.asarray(host["is_ext"])
                           & np.asarray(host["mmask"]))):
            compute_flux_marg = False

    probes: list[tuple] = []
    if _CHUNK_OVERRIDE is not None:
        # debug/probe forcing (TRACT7DT_ISLAND_CHUNK) wins over measurement
        w = max(1, int(_CHUNK_OVERRIDE))
        if log_fn is not None:
            log_fn(f"chunk width: TRACT7DT_ISLAND_CHUNK={w} overrides the "
                   f"measurement (N={N})")
        return dict(width=w, cost_bytes=0, budget_bytes=int(budget_bytes),
                    per_island_bytes=0, probes=probes,
                    wall_s=time.perf_counter() - t0, dominant="override",
                    verified=False)

    def _kernel_calls(w: int):
        """(tag, jitted fn, args) for EVERY kernel a width-w chunk calls at
        the full width, with the production argument pytree."""
        rows = np.arange(min(w, n_islands), dtype=np.int64)
        isl, p_seed, _ = _build_chunk_inputs(
            gather, addback_v, psf_tree, host, rows, w, pad,
            dstack_dev, wstack_dev)
        p = p_seed
        lam = jnp.zeros(w, jnp.float32)
        chi2 = jnp.zeros(w, jnp.float64 if kmode == "dlnp" else jnp.float32)
        i32 = jnp.zeros(w, jnp.int32)
        bfl = jnp.zeros(w, bool)
        if kmode in ("changed", "dlnp"):
            step_args = (p, lam, chi2, i32, i32, bfl, bfl, bfl, i32,
                         jnp.zeros((w, nP), jnp.float32), i32, isl, psf_tree)
        else:
            step_args = (p, lam, chi2, i32, i32, bfl, bfl, i32, isl, psf_tree)
        ro = (p, isl, psf_tree)
        # EVERY kernel this chunk will actually call at the FULL width.
        # (The stalled-subset probes -- diverged_v / plateau_v -- run at
        # power-of-two SUB-widths and are bounded by these; the OOM retry
        # backstops the pathological all-rows-stalled case.)
        todo = [("chi2_v", chi2_v, ro), ("step_v", step_v, step_args),
                ("final_v", final_v, ro)]
        if compute_param_errors:
            todo.append(("err_v", err_v, ro))
        if compute_flux_marg and fluxmarg_v is not None:
            todo.append(("fluxmarg_v", fluxmarg_v, ro))
        if compute_source_metrics:
            todo.append(("metrics_v", metrics_v, ro))
        if compute_re_profile and reprof_vk is not None:
            todo.append(("reprof_v", reprof_vk, ro))
        if want_residual:
            todo.append(("residual_v", residual_v, ro))
        return todo

    per_kernel: dict = {}

    def _probe(w: int):
        """Measured device demand of a width-w chunk (bytes), or None.

        Width bundle #2 (2026-08-15): if the caller supplied a cross-tile
        cost table entry for this width (same engine/jax/device/shape key,
        parent-checked), its recorded XLA report is used and the probe
        compile -- a pure sunk cost, its no-autotune executable is never
        reused by production -- is skipped. Disclosure: the probe tuple's
        tag carries a "+table" suffix, visible in the per-class log line.
        Safety is structural: the chosen width is still compiled for real,
        and the runtime OOM-halving retry stands; a wrong table entry can
        cost performance, never science.
        """
        if cost_table is not None:
            hit = cost_table.get(int(w))
            if hit is not None:
                c_t, who_t = int(hit[0]), str(hit[1])
                probes.append((w, c_t, who_t + "+table"))
                return c_t, who_t
        worst, who = 0, ""
        by_tag: dict = {}
        for tag, fn, a in _kernel_calls(w):
            b = _xla_call_bytes(fn, a)
            if b is None:
                probes.append((w, None, tag))
                return None, tag
            by_tag[tag] = b
            if b > worst:
                worst, who = b, tag
        per_kernel[int(w)] = by_tag
        probes.append((w, worst, who))
        return worst, who

    def _measure_production(w: int):
        """Compile the PRODUCTION executables at width ``w`` (autotuner ON,
        chunk arguments resident) and read XLA's own per-call report off
        them.

        effarc2 K13 (2026-08-26). This merges the old ``_verify`` with the
        wide-rung probe. The pre-K13 search compiled every candidate rung
        TWICE: once through :func:`_xla_call_bytes` with the autotuner off
        (a pure sunk cost -- that executable carries different compile
        options, so the persistent cache never hands it to production) and
        once for real at the end. Measured on the sparse anchor: the probe
        phase is 129 s of a 765 s cold run and 56% of all island shard-worker
        samples are ``backend_compile_and_load``; the autotune-OFF path is
        also the one whose cost scales hardest with HLO op count, which is
        why it got ~10 s/class MORE expensive when the Lanczos shift stopped
        being a single cuDNN convolution.

        Nothing is adopted unmeasured: the width that ships is the width
        whose REAL executables were compiled and reported here. Returns
        ``(bytes, dominant_tag)``, or ``(None, reason)`` when the machine
        cannot build it (a memory-class failure -- the caller steps down a
        rung; anything else re-raises).
        """
        worst, who = 0, ""
        try:
            for tag, fn, a in _kernel_calls(w):
                comp = fn.lower(*a).compile()
                m = comp.memory_analysis()
                if m is None:
                    return None, f"{tag}: no report"
                b = int(m.temp_size_in_bytes + m.argument_size_in_bytes
                        + m.output_size_in_bytes)
                if b > worst:
                    worst, who = b, tag
        except Exception as exc:                               # noqa: BLE001
            if not _is_device_mem_error(exc):
                raise
            return None, f"{type(exc).__name__}: {str(exc)[:90]}"
        probes.append((w, worst, who + "+prod"))
        return worst, who

    def _probe_kernel_at(w: int, tag: str):
        """Cheap probe of ONE kernel at width w; records it into per_kernel."""
        for t, fn, a in _kernel_calls(w):
            if t != tag:
                continue
            b = _xla_call_bytes(fn, a)
            if b is None:
                probes.append((w, None, t))
                return None, t
            per_kernel.setdefault(int(w), {})[t] = b
            # tagged "+1k" so the log and the cost table can tell this entry
            # apart: it is ONE kernel's demand, not the max over the set.
            probes.append((w, b, t + "+1k"))
            return b, t
        # the recorded dominant kernel is not in this probe set any more (K2b
        # drops fluxmarg_v for star-only classes) -- measure the whole set.
        return _probe(w)

    def _extrapolate_rung(limit_w: int, budget: int) -> "int | None":
        """Largest ladder rung whose PREDICTED demand fits, from the two
        cheap small-shape probes at rungs 1 and 2.

        Per kernel the XLA report is affine in the chunk width (a fixed part
        -- the psf tree, the per-class constants -- plus a per-island part);
        rungs 1 and 2 determine both, and the maximum over kernels is the
        chunk's demand. This only chooses which rung to COMPILE FOR REAL:
        :func:`_measure_production` then measures that rung on the real
        executables and the caller steps down if the measurement (or the
        compile) says no. Returns None when the two points are unusable, and
        the caller falls back to the pre-K13 rung-by-rung walk.
        """
        k1, k2 = per_kernel.get(1), per_kernel.get(2)
        if not k1 or not k2:
            return None
        # slope/intercept of the ONE kernel measured at both rungs; the others
        # ride at their rung-1 ratio (they are the same per-island batched
        # shape family, and the production measurement is the arbiter).
        ref = next(iter(k2))
        if ref not in k1 or k2[ref] <= k1[ref]:
            return None
        a_ref = max(2 * k1[ref] - k2[ref], 0)
        b_ref = k2[ref] - k1[ref]
        w = 1
        while w * 2 <= int(limit_w):
            nxt = w * 2
            pred = max((a_ref + b_ref * nxt) * (k1[t] / max(k1[ref], 1))
                       for t in k1)
            if pred > budget:
                break
            w = nxt
        return int(w)


    c1, dom = _probe(1)
    if c1 is None or c1 > budget_bytes:
        if log_fn is not None:
            log_fn(f"chunk width: bbox N={N} does not fit one island "
                   f"({'unmeasurable' if c1 is None else f'{c1 / 2 ** 30:.2f} GiB'}"
                   f" vs {budget_bytes / 2 ** 30:.2f} GiB free) -> giant "
                   f"path (seed geometry)")
        return dict(width=0, cost_bytes=(c1 or 0), budget_bytes=budget_bytes,
                    per_island_bytes=(c1 or 0), probes=probes,
                    wall_s=time.perf_counter() - t0, dominant=dom,
                    verified=False)

    # T6/I1 (2026-08-23, development ledger SS21): the probe oracle is
    # memoised per rung so the class-level extension below never compiles a
    # rung twice, and the tile-level search is the unchanged pure walk.
    _probe_memo: dict = {}

    def _probe_once(w: int):
        w = int(w)
        if w not in _probe_memo:
            _probe_memo[w] = _probe(w)
        return _probe_memo[w]

    # K13: two cheap small-shape points (rungs 1 and 2) fix the affine cost
    # model per kernel; the model picks ONE rung to compile for real.
    best_w = best_c = None
    best_dom = dom
    _verified_w = None          # width whose PRODUCTION build fitted the budget
    if int(limit) >= 2 and per_kernel.get(1) is not None:
        # `per_kernel[1]` is absent when the rung-1 cost came from the
        # cross-tile cost table (that branch returns before recording the
        # per-kernel split). Without it the affine model cannot be built, so
        # skip straight to the pre-K13 walk instead of paying a rung-2 probe
        # whose result would be discarded.
        # Only the rung-1 ARGMAX kernel is re-probed at rung 2. A cheap probe
        # costs one XLA compile, and compile time is set by the PROGRAM, not
        # by the width -- measured: a 6-kernel probe set costs ~15 s at width
        # 1 just as at width 2048 -- so probing all six twice is the expensive
        # half of the search. One kernel's two points give its slope; the
        # others are carried at their rung-1 ratio. This only CHOOSES the rung
        # to build for real: `_measure_production` then measures every kernel
        # on the real executables and steps down if the truth disagrees.
        c2, dom2 = _probe_kernel_at(2, dom)
        if c2 is not None:
            cand = _extrapolate_rung(int(limit), int(budget_bytes))
            if cand is not None:
                w_try = int(cand)
                while w_try >= 1:
                    mb, mdom = _measure_production(w_try)
                    if mb is not None and mb <= int(budget_bytes):
                        best_w, best_c, best_dom = w_try, mb, mdom
                        _verified_w = w_try
                        break
                    if log_fn is not None:
                        log_fn(f"chunk width: N={N} width {w_try} "
                               f"{'did not build' if mb is None else 'measured over budget'}"
                               f" ({mdom}) -- stepping down")
                    if w_try == 1:
                        break
                    w_try //= 2
    if best_w is None:
        # fallback: the pre-K13 rung-by-rung walk on the cheap probe
        best_w, best_c, best_dom = ladder_walk(
            _probe_once, int(limit), int(budget_bytes), c1, dom)
    # T6/I1: the memory rung is a property of the CLASS (kernel shapes x this
    # card), not of the tile, but the walk above stops at the tile's
    # island-count rung, so the cross-tile cost table only ever learned the
    # rungs small tiles happen to visit (S0-3: neighbours hit 13/15, a tile
    # with different island counts hit 0/20 and re-probed everything).
    # Walk the same ladder once more to the ceiling -- memoised, so only
    # rungs the tile walk did not visit are probed -- purely to fill the
    # table; the PRODUCTION width stays the tile walk's result above
    # (unchanged decision, unchanged executables). Later tiles of the
    # family then find every rung their own walk visits in the table and
    # probe nothing. First encounter of a class pays the extra rungs once
    # per card; nothing is extrapolated (every table entry is a measured
    # rung); and the table is only consulted when a cache dir exists.
    n_class_extra = 0
    if class_ladder and int(limit) < int(max_width) and best_w >= 1:
        _n0 = len(probes)
        ladder_walk(_probe_once, int(max_width), int(budget_bytes), c1, dom)
        n_class_extra = len(probes) - _n0

    # phase 2: can this machine actually BUILD it (autotuner scratch and
    # all)? Step down a rung per failure. K13: when the width came from
    # _measure_production the production executables ARE already built and
    # reported, so this loop is a no-op for it; it still guards the fallback
    # walk's result.
    verified = not verify_compile
    if verify_compile and _verified_w is not None and best_w == _verified_w:
        # the primary K13 loop already compiled the production executables at
        # this width AND measured them inside the budget
        verified = True
    while verify_compile and not verified and best_w >= 1:
        mb, why = _measure_production(best_w)
        if mb is not None and mb <= int(budget_bytes):
            verified = True
            best_c = mb
            best_dom = why or best_dom
            break
        if mb is not None:
            why = (f"measured {mb / 2 ** 30:.2f} GiB > budget "
                   f"{budget_bytes / 2 ** 30:.2f} GiB")
        if log_fn is not None:
            log_fn(f"chunk width: N={N} width {best_w} passed the memory "
                   f"measurement but would not COMPILE ({why}) -- the "
                   f"autotuner's scratch does not fit; stepping down")
        if best_w == 1:
            break
        best_w //= 2
        c, dm = _probe(best_w)
        best_c = c if c is not None else best_c
        best_dom = dm or best_dom

    # ---- rung-boundary disclosure (2026-08-30, development ledger SS24) ----
    # The chosen width is the largest LADDER RUNG whose measured cost fits a
    # budget sampled from the card's CURRENT free memory. That makes the
    # decision co-residency dependent, and the chunk width is a documented
    # science channel in this engine (SS23-18 measured 6.50 mmag from changing
    # only the width). How close the decision came to flipping was not
    # recorded anywhere, and the log's `%.2f GiB` cannot show it: on T08285,
    # class (bbox=32, nsrc=32, star) rejected width 512 at 12,485 MiB against
    # a 12,472 MiB budget -- a 13 MiB / 0.10% margin that prints as
    # "12.19 GiB > 12.18 GiB". A future width flip would look exactly like a
    # science change with nothing in the record to say otherwise.
    #
    # Disclosure only: nothing below feeds a decision, so no constant is
    # introduced and no behaviour changes. `next_rung_bytes` is None when the
    # next rung's probe FAILED to build -- reported as such rather than
    # silently omitted, because that is the case a reader most needs.
    _next_w = int(best_w) * 2
    _next_b = None
    for _w, _b, _ in probes:
        if int(_w) == _next_w:
            _next_b = _b
            break
    out = dict(width=int(best_w), cost_bytes=int(best_c),
               budget_bytes=int(budget_bytes),
               per_island_bytes=int(best_c / max(best_w, 1)),
               probes=probes, wall_s=time.perf_counter() - t0,
               dominant=best_dom, verified=bool(verified),
               next_rung_width=_next_w,
               next_rung_bytes=(None if _next_b is None else int(_next_b)),
               next_rung_margin_bytes=(None if _next_b is None
                                       else int(budget_bytes) - int(_next_b)),
               margin_bytes=int(budget_bytes) - int(best_c))
    if log_fn is not None:
        if _next_b is None:
            _nxt = f"next rung {_next_w}: not probed/failed to build"
        else:
            _nxt = (f"next rung {_next_w}: {_next_b} B, margin "
                    f"{int(budget_bytes) - int(_next_b)} B "
                    f"({100.0 * (int(budget_bytes) - int(_next_b)) / max(int(budget_bytes), 1):+.2f}%)")
        log_fn(
            f"chunk width (measured): N={N} -> {best_w} island(s)/chunk; "
            f"{best_c / 2 ** 30:.2f} GiB needed by {best_dom} of "
            f"{budget_bytes / 2 ** 30:.2f} GiB free; "
            f"{len(probes)} probe(s) in {out['wall_s']:.1f}s "
            f"[{', '.join(f'{w}:{(b / 2 ** 20):.0f}MiB' if b else f'{w}:FAIL' for w, b, _ in probes)}]"
            + (f"; class ladder: +{n_class_extra} rung(s) measured for the "
               f"cross-tile table (width unchanged)" if n_class_extra else "")
            + f"; budget {int(budget_bytes)} B exact, chosen margin "
              f"{int(budget_bytes) - int(best_c)} B, {_nxt}")
    return out


_COUNT_GOOD_JIT = None


def _count_good(w):
    """Per-island count of unmasked pixels (summed over bands), compiled."""
    global _COUNT_GOOD_JIT
    if _COUNT_GOOD_JIT is None:
        import jax
        import jax.numpy as jnp
        _COUNT_GOOD_JIT = jax.jit(
            lambda a: jnp.sum(a > 0, axis=(1, 2, 3)))
    return _COUNT_GOOD_JIT(w)


def _build_chunk_inputs(gather, addback_v, psf_tree, host, rows, width, pad,
                        dstack_dev, wstack_dev):
    """Gather ONE chunk's device inputs: the ``isl`` argument pytree, the
    flat seed parameter vector, and the T5 probe's bucket-identity string.

    Factored out of :func:`run_class_chunk` so the T5/#28 chunk-width
    MEASUREMENT (:func:`measure_class_chunk_width`) lowers EXACTLY the
    production argument pytree. A hand-written copy of these 17 leaves
    would drift silently, and a drifted pytree measures a DIFFERENT program
    than the one that runs -- which is the class of mistake that put the
    engine in this position (the byte model was calibrated through a path
    that could not even run ``fluxmarg_v``). Pure input construction: no LM
    state, and no kernel calls except the two window gathers and the
    scene-subtract member add-back that DEFINES the cutout.
    """
    import jax.numpy as jnp

    N = int(host["mmask"].shape[1])
    nP = 6 * N
    n_c = int(rows.size)
    width = int(width)
    pad_n = width - n_c

    def _padn(a, fill=0):
        if pad_n == 0:
            return a
        return np.concatenate(
            [a, np.full((pad_n,) + a.shape[1:], fill, dtype=a.dtype)],
            axis=0)

    d_c = gather(dstack_dev, _padn(host["oy"][rows], pad),
                 _padn(host["ox"][rows], pad))
    w_c = gather(wstack_dev, _padn(host["oy"][rows], pad),
                 _padn(host["ox"][rows], pad))
    # T5/#28 probe identity (formatted only when the probe is on).
    ident = ""
    if _MEMPROBE:
        ident = (f"B={int(d_c.shape[-1])} N={N} "
                 f"kind={'mixed' if bool(host['is_ext'].any()) else 'star'} "
                 f"t={int(psf_tree['coefs'].shape[1])}")
    isl = {
        "data": d_c, "w": w_c,
        "x0": jnp.asarray(_padn(host["x0b"][rows])),
        "y0": jnp.asarray(_padn(host["y0b"][rows])),
        "stype": jnp.asarray(_padn(host["st"][rows])),
        "n_frozen": jnp.asarray(_padn(host["sn"][rows], 1.0)),
        "free_n": jnp.asarray(_padn(host["fn_flag"][rows], False)),
        "mmask": jnp.asarray(_padn(host["mmask"][rows], False)),
        # PERF #14: galaxy-lane gather map (empty for star classes; unused
        # when star_render_split is off). The mixed split kernel scatters
        # the G FFT galaxy renders into these slots. Padded islands get
        # gsel=0 / gvalid=False -> the mixed scatter is a same-value no-op
        # there and the island is zeroed by mmask.
        "gsel": jnp.asarray(_padn(host["gsel"][rows], 0)),
        "gvalid": jnp.asarray(_padn(host["gvalid"][rows], False)),
        "keep": jnp.asarray(_padn(host["keep"][rows], False)),
        "lower": jnp.asarray(
            _padn(host["lower"][rows], -1.0).reshape(width, nP)),
        "upper": jnp.asarray(
            _padn(host["upper"][rows], 1.0).reshape(width, nP)),
    }
    if "cb_scale" in host:
        # F3-S2: per-island crossband amplitude scale (crossband mode only;
        # absent in band mode -> pytree/program unchanged there).
        isl["cb_scale"] = jnp.asarray(_padn(host["cb_scale"][rows], 1.0))
    # Catalog-driven NONLINEAR prior leaves (position/shape/Sersic-n;
    # always present -- the kernel only reads them when its static
    # has_priors flag is set, else JAX DCE's them so the compiled program
    # is byte-identical to the pre-prior kernel). prior mean = seed
    # (host["prior_mean"] == host["p0"]); inv_sigma == 0 where no prior
    # was supplied. Padding contributes inv_sigma = 0. (Per-band FLUX
    # priors are DEFERRED to the P7 field-level flux solve.)
    isl["prior_mean"] = jnp.asarray(
        _padn(host["prior_mean"][rows], 0.0).reshape(width, nP))
    isl["prior_inv_sigma"] = jnp.asarray(
        _padn(host["prior_inv_sigma"][rows], 0.0).reshape(width, nP))
    p_seed_flat = jnp.asarray(_padn(host["p0"][rows]).reshape(width, nP))
    # #35 scene-subtract: add this island's members back to the field-wide
    # scene-subtracted cutout, at their S1 flux + SEED shape/pos, so the joint
    # fit sees members at TRUE data level while NON-members stay removed. Uses
    # the fixed p0 seed (independent of init_state / re-queue carry: the field-
    # wide pass subtracted members at their seed, so the cancellation is at the
    # seed), and re-applies on EVERY chunk call since each gather re-reads the
    # scene-subtracted device stack. member_s1_flux built by class_host_arrays.
    if addback_v is not None:
        isl["member_s1_flux"] = jnp.asarray(
            _padn(host["member_s1_flux"][rows], 0.0))
        isl["data"] = isl["data"] + addback_v(p_seed_flat, isl, psf_tree)
    return isl, p_seed_flat, ident


def run_class_chunk(
    kernels: tuple,
    psf_tree: dict,
    host: dict,                     # island_plan.class_host_arrays output
    rows: np.ndarray,               # (n_c,) indices into the class arrays
    width: int,                     # vmap batch width (>= n_c; padded)
    max_steps: int,
    dstack_dev,                     # (n_bands, Hp, Wp) padded device stack
    wstack_dev,
    pad: int,
    init_state: dict | None = None,
    compute_param_errors: bool = False,
    want_residual: bool = False,
    compute_source_metrics: bool = False,
    compute_flux_marg: bool = False,
    compute_re_profile: bool = False,   # #19a Re profile scan
) -> dict:
    """Fit ONE chunk of a class's islands (the multi-GPU scheduling unit).

    ``init_state`` (dict of per-row ``p/lam/chi2/consec/nrej/nst``
    arrays) continues re-queued islands EXACTLY where their main pass
    stopped -- the LM trajectory is the same math as the single-pass
    solver; only the vmap batch width differs (ulp-level fp
    reassociation, the documented chunk-width noise class). Fresh
    chunks (``init_state=None``) start from the catalog seeds in
    ``host``.

    ``compute_param_errors`` (task #28a): when True, one extra
    read-only ``err_v`` evaluation at the converged state returns
    ``perr (n_c, N, 13)`` -- the marginal nonlinear-parameter variances
    ``[var_x, var_y, var_logre, var_ee1, var_ee2, var_un,
    cov_ee1_ee2]`` followed by the 6 per-internal-param prior fractions
    (see ``final_errors``). When False (default) the kernel is not
    called and ``perr`` is None; all other outputs are identical either
    way. Re-queued (phase-B) islands overwrite their phase-A rows, so
    requeues need no special handling.

    Returns per-row numpy arrays (sliced back to ``n_c``):
    ``p (n_c,N,6), fluxes/ferr (n_c,n_bands,N), chi2, conv, stall, nst,
    lam, consec, nrej, perr`` + ``wall_s`` and ``steps_run``. Under the
    "dlnp" step protocol the ``chi2`` state (init, carry, output) is
    FLOAT64 -- the f64 carry survives the A->B requeue via
    ``init_state`` untouched. #35 adds
    ``diverged (n_c,) bool`` (frozen-proposal-stall disclosure: POST-HOC
    ``diverged_v`` reconstruction on the stalled subset in legacy mode,
    the in-kernel sticky flag in changed/dlnp modes), ``n_same (n_c,) int32``
    + ``prej (n_c, 6N) float32`` (the changed-mode tracker carry;
    zeros in legacy mode), and -- when ``compute_source_metrics`` --
    ``swin (n_c, n_bands, N)``, the per-band data-constrained window
    fluxes (FLUX_WIN). ``init_state`` dicts without the #35 keys (e.g.
    the #34 0-step residual-export path) default them to zeros.
    """
    import jax.numpy as jnp

    t0 = time.perf_counter()
    (gather, step_v, chi2_v, final_v, err_v, residual_v,
     metrics_v) = kernels[:7]
    # #35: static step protocol recorded by _make_bucket_kernels
    # ("proposals" = legacy byte-identical step + post-hoc diverged_v
    # probe on the stalled subset; "changed" = 11-state in-kernel
    # tracker).
    diverged_v = kernels[7] if len(kernels) > 8 else None
    kmode = kernels[8] if len(kernels) > 8 else "proposals"
    # D2: SHAPE-marginalized flux-error increment kernel (appended to the
    # tuple after kmode). Older cached tuples without it -> None (feature
    # unavailable; compute_flux_marg stays a no-op).
    fluxmarg_v = kernels[9] if len(kernels) > 9 else None
    reprof_vk = kernels[13] if len(kernels) > 13 else None
    plateau_vk = kernels[14] if len(kernels) > 14 else None
    # dlnp (E1a): the statistical quantum for the reject-exit mapping
    # (None for legacy/changed tuples -> mapping disabled).
    dlnp_quantum = kernels[15] if len(kernels) > 15 else None
    # Trust-region (step_mode="trust_region"): railed_v = bound-pinned
    # disclosure probe (appended after fluxmarg_v); smode = static step mode.
    # Older cached tuples (pre-trust-region) -> None / "clip".
    railed_v = kernels[10] if len(kernels) > 10 else None
    smode = kernels[11] if len(kernels) > 11 else "clip"
    # #35 scene-subtract: member add-back kernel (None unless scene_subtract).
    addback_v = kernels[12] if len(kernels) > 12 else None
    N = int(host["mmask"].shape[1])
    nP = 6 * N
    n_c = int(rows.size)
    width = int(width)
    if width < n_c:
        raise ValueError(f"chunk width {width} < rows {n_c}")
    pad_n = width - n_c

    def _padn(a, fill=0):
        if pad_n == 0:
            return a
        return np.concatenate(
            [a, np.full((pad_n,) + a.shape[1:], fill, dtype=a.dtype)],
            axis=0)

    isl, p_seed_flat, _ident = _build_chunk_inputs(
        gather, addback_v, psf_tree, host, rows, width, pad,
        dstack_dev, wstack_dev)
    if init_state is None:
        p = p_seed_flat
        # trust-region carries the radius Delta in the legacy lam slot; the
        # 0.0 "unset" sentinel makes one_step_tr seed Delta0 = the first GN
        # step magnitude (parameter-free). "clip" keeps LM_LAMBDA0.
        lam = jnp.full(width,
                       0.0 if smode == "trust_region" else LM_LAMBDA0,
                       dtype=jnp.float32)
        chi2 = chi2_v(p, isl, psf_tree)
        consec = jnp.zeros(width, dtype=jnp.int32)
        nrej = jnp.zeros(width, dtype=jnp.int32)
        nst = jnp.zeros(width, dtype=jnp.int32)
    else:
        p = jnp.asarray(_padn(
            init_state["p"].astype(np.float32)).reshape(width, nP))
        lam = jnp.asarray(_padn(init_state["lam"].astype(np.float32),
                                LM_LAMBDA0))
        # dlnp mode: the chi2 carry is FLOAT64 across the A->B requeue
        # boundary too -- a silent f32 downcast here would reintroduce
        # the fp32 accept coin-flip on exactly the re-queued (hard)
        # islands.
        _chi2_dt = np.float64 if kmode == "dlnp" else np.float32
        chi2 = jnp.asarray(_padn(init_state["chi2"].astype(_chi2_dt)))
        consec = jnp.asarray(_padn(init_state["consec"].astype(np.int32)))
        nrej = jnp.asarray(_padn(init_state["nrej"].astype(np.int32)))
        nst = jnp.asarray(_padn(init_state["nst"].astype(np.int32)))
    # #35 tracker state (HOST numpy; moved on-device only in changed
    # mode). init_state dicts without the keys (e.g. the #34 0-step
    # residual-export init) default to zeros.
    if init_state is None:
        div_h = np.zeros(width, dtype=bool)
        n_same_h = np.zeros(width, dtype=np.int32)
        prej_h = np.zeros((width, nP), dtype=np.float32)
    else:
        div_h = _padn(np.asarray(init_state.get(
            "diverged", np.zeros(n_c, dtype=bool))).astype(bool))
        n_same_h = _padn(np.asarray(init_state.get(
            "n_same", np.zeros(n_c, dtype=np.int32))).astype(np.int32))
        prej_h = _padn(np.asarray(init_state.get(
            "prej", np.zeros((n_c, nP), dtype=np.float32))).astype(
            np.float32)).reshape(width, nP)
    conv = jnp.asarray(_padn(np.zeros(n_c, dtype=bool), True))
    stall = jnp.zeros(width, dtype=bool)

    steps_run = 0
    pexc_h = np.full(width, np.nan, dtype=np.float32)
    if kmode in ("changed", "dlnp"):
        # in-kernel tracker (the "changed" counting rule needs the
        # proposal-identity signal inside the step program; "dlnp"
        # shares the 11-state layout with the f64 chi2 carry).
        div = jnp.asarray(div_h)
        n_same = jnp.asarray(n_same_h)
        prej = jnp.asarray(prej_h)
        for _it in range(int(max_steps)):
            (p, lam, chi2, consec, nrej, conv, stall, div, n_same, prej,
             nst) = step_v(p, lam, chi2, consec, nrej, conv, stall, div,
                           n_same, prej, nst, isl, psf_tree)
            steps_run += 1
            if bool(jnp.all(conv)):
                break
        div_h = np.asarray(div)
        n_same_h = np.asarray(n_same)
        prej_h = np.asarray(prej)
        # Stall-yield (2026-07-27): the plateau certificate is
        # mode-independent (no reject-history reconstruction needed).
        _rows_st = np.nonzero(np.asarray(stall)[:n_c])[0]
        if _rows_st.size and plateau_vk is not None:
            wst = 1 << (int(_rows_st.size) - 1).bit_length()
            sel = np.concatenate([
                _rows_st,
                np.zeros(wst - _rows_st.size, dtype=np.int64)])
            sel_d = jnp.asarray(sel, dtype=jnp.int32)
            isl_st = {k: v[sel_d] for k, v in isl.items()}
            psub = plateau_vk(p[sel_d], nst[sel_d], isl_st, psf_tree)
            pexc_h[_rows_st] = np.asarray(psub)[:_rows_st.size]
    else:
        # legacy step: the compiled program is BYTE-IDENTICAL to the
        # pre-#35 solver (same jaxpr -> same persistent-JIT-cache entry).
        _memprobe("step_v", step_v,
                  (p, lam, chi2, consec, nrej, conv, stall, nst, isl,
                   psf_tree), width, _ident)
        for _it in range(int(max_steps)):
            p, lam, chi2, consec, nrej, conv, stall, nst = step_v(
                p, lam, chi2, consec, nrej, conv, stall, nst, isl,
                psf_tree)
            steps_run += 1
            if bool(jnp.all(conv)):
                break
        # #35 POST-HOC diverged reconstruction, read-only, on the
        # STALLED subset only (a stall is exactly K consecutive rejects
        # at the FINAL p with lam escalated x10 each -- the K tested
        # proposals are a pure function of the final state; see
        # diverged_probe). Sub-batch padded to a power of two so the
        # probe compiles for O(log width) widths per class.
        stall_np = np.asarray(stall)[:n_c]
        rows_st = np.nonzero(stall_np)[0]
        # effarc2 G2g (2026-08-26): build the padded stalled-subset gather
        # ONCE. Both read-only probes below take the same rows at the same
        # power-of-two sub-width, and `isl_st` is a 17-leaf device gather that
        # includes the (wst, n_bands, B, B) data and weight cubes -- doing it
        # twice cost a second full copy of them (~46 MB sparse, ~200 MB
        # crowded per chunk) at a moment the measured width budget never
        # accounted for. Same values, same order: bit-identical.
        isl_st = sel_d = None
        if rows_st.size and (diverged_v is not None or plateau_vk is not None):
            wst = 1 << (int(rows_st.size) - 1).bit_length()
            sel = np.concatenate([
                rows_st,
                np.zeros(wst - rows_st.size, dtype=np.int64)])
            sel_d = jnp.asarray(sel, dtype=jnp.int32)
            isl_st = {k: v[sel_d] for k, v in isl.items()}
        if rows_st.size and diverged_v is not None:
            dsub = diverged_v(p[sel_d], lam[sel_d], nst[sel_d],
                              isl_st, psf_tree)
            div_h[rows_st] = np.asarray(dsub)[:rows_st.size]
        # Stall-yield (2026-07-27): probability-convergence certificate
        # on the same stalled subset (read-only; mode-independent).
        if rows_st.size and plateau_vk is not None:
            psub = plateau_vk(p[sel_d], nst[sel_d], isl_st, psf_tree)
            pexc_h[rows_st] = np.asarray(psub)[:rows_st.size]

    # dlnp exit-state mapping (E1a, internal dlnp design note §2e): a
    # reject-exit (K distinct rejects / frozen-at-cap / no-move) whose
    # full-ladder certificate PASSES (best available damped-step dchi2
    # within the statistical quantum) is CONVERGED-ON-PLATEAU -- the
    # island stopped because it is DONE (measured on the UDS arm:
    # 210/220 reject-exits certify at dchi2 ~ 0 with endpoints equal to
    # legacy's). Only certificate-FAILING reject-exits stay stalled
    # (descent available but unreachable -- the honest stall; hard-fail
    # semantics unchanged). island_plateau is derived downstream from
    # the finite stall_dchi2 (pipeline.py, mode-aware). conv is already
    # True either way (stall is a convergence prong).
    stall_h = np.asarray(stall)[:n_c]
    if kmode == "dlnp" and dlnp_quantum is not None:
        stall_h = stall_h.copy()
        cert = (stall_h & np.isfinite(pexc_h[:n_c])
                & (pexc_h[:n_c] > -float(dlnp_quantum)))
        stall_h[cert] = False

    # Trust-region: bound-pinned disclosure on ALL converged rows (cheap --
    # a param-vs-bound comparison, no fit). Replaces the (now-vacuous under
    # the dogleg) clip-saturation diverged probe. All-False in clip mode.
    railed_h = np.zeros(n_c, dtype=bool)
    if smode == "trust_region" and railed_v is not None:
        railed_h = np.asarray(railed_v(p, isl))[:n_c].astype(bool)

    _memprobe("final_v", final_v, (p, isl, psf_tree), width, _ident)
    f_c, fe_c = final_v(p, isl, psf_tree)
    perr = None
    if compute_param_errors:
        _memprobe("err_v", err_v, (p, isl, psf_tree), width, _ident)
        perr = np.asarray(err_v(p, isl, psf_tree))[:n_c]
    # D2: SHAPE-marginalized flux-error INCREMENT (n_c, n_bands, N). One
    # extra read-only Hessian-inverse + VarPro-Jacobian evaluation at the
    # converged state; gated so an off run is byte-identical (kernel not
    # called). 0 for star rows / padded slots.
    femarg = None
    if compute_flux_marg and fluxmarg_v is not None:
        if not bool(np.any(np.asarray(host["is_ext"])[rows]
                           & np.asarray(host["mmask"])[rows])):
            # effarc2 K2 (2026-08-26): final_flux_marg masks every column
            # AND every row by ``is_ext`` (a point source has no free
            # shape block, so its increment is EXACTLY 0.0 by
            # construction -- see the kernel's MARGINALIZATION SET note).
            # A chunk with no extended member therefore returns an
            # all-zero (n_c, n_bands, N) array; produce it without the
            # kernel (measured 27-31% of every star-class chunk: one
            # hg_scan + an N x nP jacrev + an nP x nP f64 inverse, all
            # discarded). Bit-identical: the host forms
            # FLUXERR_MARG = sqrt(FLUXERR^2 + 0). Mixed chunks (any
            # extended member) run the kernel unchanged.
            femarg = np.zeros((n_c, int(isl["data"].shape[1]), N),
                              dtype=np.float32)
        else:
            # THE kernel that asked for 19.68 GiB and killed T08285 on
            # 2026-08-02, and the one the chunk-width byte model has never
            # seen (see _memprobe).
            _memprobe("fluxmarg_v", fluxmarg_v, (p, isl, psf_tree), width,
                      _ident)
            femarg = np.asarray(fluxmarg_v(p, isl, psf_tree))[:n_c]
    # #19a: Re profile-likelihood scan at the converged state.
    reprof = None
    if compute_re_profile and reprof_vk is not None:
        reprof = np.asarray(reprof_vk(p, isl, psf_tree))[:n_c]
    resid = resid_w = None
    if want_residual:
        # Task #34: per-island coadded model-subtracted residual + weight
        # at the converged params (one extra render pass; only requested on
        # the re-detect path). Sliced back to the real island count.
        Rn, Wc = residual_v(p, isl, psf_tree)
        resid = np.asarray(Rn)[:n_c]
        resid_w = np.asarray(Wc)[:n_c]
    smetrics = swin = None
    if compute_source_metrics:
        # Task #17: per-source [rchisq, fracflux, fracin, fracmasked] at the
        # converged params (one extra read-only band scan; gated so an off run
        # is byte-identical). Sliced back to the real island count. #35: the
        # same scan also yields the per-band data-constrained window fluxes
        # (FLUX_WIN) -- swin (n_c, n_bands, N).
        _memprobe("metrics_v", metrics_v, (p, isl, psf_tree), width,
                  _ident)
        sm, sw = metrics_v(p, isl, psf_tree)
        smetrics = np.asarray(sm)[:n_c]                       # (n_c, N, 4)
        swin = np.asarray(sw)[:n_c]                           # (n_c, nb, N)
    # LEG D nonlinear (task #20): # of good (unmasked) pixels per island,
    # summed over bands -- the numerator of dof for the per-island reduced
    # chi2 (chi2_nu = chi2 / (n_good - n_free)). Cheap device reduction on
    # the gathered weight; the whitened residual/chi2 uses the SAME w>0
    # pixels, so this is the exact pixel count entering chi2.
    # effarc2 G3g (2026-08-26): the good-pixel count ran as TWO EAGER kernels
    # over the whole (width, n_bands, B, B) weight cube (369 MB at (32,1)
    # w=4096) plus a host sync, and -- like every eager op -- compiled a fresh
    # program for each new width. One cached jit per shape instead. Sum of
    # booleans is exact, so this is bit-identical.
    n_good = np.asarray(_count_good(isl["w"]), dtype=np.int64)[:n_c]
    return dict(
        p=np.asarray(p).reshape(width, N, 6)[:n_c],
        fluxes=np.asarray(f_c)[:n_c],
        ferr=np.asarray(fe_c)[:n_c],
        perr=perr,
        femarg=femarg,
        reprof=reprof,
        smetrics=smetrics,
        swin=swin,
        resid=resid,
        resid_w=resid_w,
        n_good=n_good,
        chi2=np.asarray(chi2)[:n_c],
        conv=np.asarray(conv)[:n_c],
        stall=stall_h,
        diverged=div_h[:n_c],
        stall_dchi2=pexc_h[:n_c],
        railed=railed_h,
        n_same=n_same_h[:n_c],
        prej=prej_h[:n_c],
        nst=np.asarray(nst)[:n_c],
        lam=np.asarray(lam)[:n_c],
        consec=np.asarray(consec)[:n_c],
        nrej=np.asarray(nrej)[:n_c],
        wall_s=time.perf_counter() - t0,
        steps_run=steps_run,
    )


def _is_device_mem_error(exc: BaseException) -> bool:
    """True for the two ways a too-wide chunk fails on the device.

    Both were observed on T08285 2026-08-02 from the SAME cause (the byte
    model sized a chunk the card could not hold): a plain
    ``RESOURCE_EXHAUSTED`` allocation failure, and -- with autotuning on --
    a compile-time ``Autotuning failed ... No valid config found`` on the
    over-large dot. Anything else propagates: a retry must not paper over
    a real bug.
    """
    e = f"{type(exc).__name__}: {exc}".upper()
    return any(k in e for k in (
        "RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OUT_OF_MEMORY",
        "NO VALID CONFIG FOUND", "AUTOTUNING FAILED"))


def _merge_chunk_outputs(parts: list) -> dict:
    """Row-concatenate the per-chunk result dicts of a split retry."""
    out = dict(parts[0])
    for k, v0 in parts[0].items():
        if k in ("wall_s", "steps_run"):
            continue
        if v0 is None:
            out[k] = None
        elif isinstance(v0, np.ndarray):
            out[k] = np.concatenate([p[k] for p in parts], axis=0)
    out["wall_s"] = float(sum(p["wall_s"] for p in parts))
    out["steps_run"] = int(max(p["steps_run"] for p in parts))
    return out


def run_class_chunk_resilient(kernels, psf_tree, host, rows, width, max_steps,
                              dstack_dev, wstack_dev, pad, *,
                              init_state=None, log_fn=None, **kw) -> dict:
    """:func:`run_class_chunk`, halving the vmap width on a device-memory
    failure instead of killing the field.

    The measured width (:func:`measure_class_chunk_width`) is XLA's own
    demand against the card's free memory, but those are not read from the
    same meter -- allocator granularity, fragmentation and the autotuner's
    own scratch live outside XLA's report. So the measurement gets the
    width close and THIS makes it safe: on a memory failure the chunk is
    re-run as two half-width chunks (recursively), which is a LADDER width
    again, so the retry compiles nothing new. Splitting changes the vmap
    batch width and therefore the fp reassociation (the documented
    chunk-width noise class) -- it is logged for exactly that reason.
    """
    try:
        out = run_class_chunk(
            kernels, psf_tree, host, rows, width, max_steps,
            dstack_dev, wstack_dev, pad, init_state=init_state, **kw)
        # effarc2: tell the caller which width was actually BUILT and RUN, so
        # a width that only ever failed here can never become a reuse target.
        out["widths_ran"] = [int(width)]
        # ...and whether this result is a SINGLE clean run. `widths_ran` is a
        # deduplicated set below, so a uniform halving collapses to one element
        # and cannot be distinguished from this branch by its length -- while
        # `wall_s` down there is the SUM over leaves. A cost model that divides
        # wall by ONE chunk's lanes needs to know the difference.
        out["halved"] = False
        return out
    except Exception as exc:                                   # noqa: BLE001
        n_c = int(np.asarray(rows).size)
        # A chunk may now be PADDED (width > n_c, effarc2 K12), so "one row
        # left" no longer implies "width 1" -- halving the WIDTH is still a
        # valid ladder retry there. Only a genuine width-1 chunk is
        # un-retryable.
        if not _is_device_mem_error(exc) or int(width) <= 1:
            raise
        if n_c <= 1 and int(width) <= 1:
            raise
        half = max(1, int(width) // 2)
        msg = (f"island chunk of {n_c} at width {width} hit "
               f"{type(exc).__name__} ({str(exc)[:80]}) -- retrying as "
               f"{int(np.ceil(n_c / half))} chunk(s) of width {half}")
        if log_fn is not None:
            log_fn(msg)
        else:
            print("island: " + msg, flush=True)
        parts = []
        for lo in range(0, n_c, half):
            sl = slice(lo, min(lo + half, n_c))
            sub_init = (None if init_state is None
                        else {k: v[sl] for k, v in init_state.items()})
            parts.append(run_class_chunk_resilient(
                kernels, psf_tree, host, np.asarray(rows)[sl], half,
                max_steps, dstack_dev, wstack_dev, pad,
                init_state=sub_init, log_fn=log_fn, **kw))
        merged = _merge_chunk_outputs(parts)
        ran: list = []
        for _p in parts:
            ran.extend(_p.get("widths_ran", []) or [])
        merged["widths_ran"] = sorted(set(int(w) for w in ran))
        # `_merge_chunk_outputs` summed `wall_s` over the leaves, so this
        # result is NOT a per-chunk cost sample at any single width.
        merged["halved"] = True
        return merged


def island_fit(
    *,
    bandnames: list,
    data_minus_bg: np.ndarray,      # (n_bands, H, W) float32, S1 bg removed
    invvar: np.ndarray,             # (n_bands, H, W) float32, bad px -> 0
    psfs: list,                     # numpy-side PsfPolynomial per band
    island_set,                     # islands.IslandSet
    buckets,                        # islands.BucketedIslands
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    source_type: np.ndarray,
    logre: np.ndarray,
    ee1: np.ndarray,
    ee2: np.ndarray,
    sersic_n: np.ndarray,
    s1_flux: np.ndarray,            # (n_src, n_bands) S1 fluxes
    s1_flux_var: np.ndarray,        # (n_src, n_bands)
    pos_err: np.ndarray | None = None,          # (n_src,) POS_ERR prior sigma (px)
    re_prior_sig: np.ndarray | None = None,     # (n_src,) Re_PRIOR_SIG (px)
    ell_prior_sig: np.ndarray | None = None,    # (n_src,) ELL_PRIOR_SIG
    theta_prior_sig: np.ndarray | None = None,  # (n_src,) THETA_PRIOR_SIG (deg)
    sersic_n_prior_sig: np.ndarray | None = None,  # (n_src,) SERSIC_N_PRIOR_SIG
    render_backend: str = "fft",
    fit_sersic_n: bool = True,
    n_fit_min_snr: float = 30.0,
    pos_max_shift_pix: float = 5.0,
    pos_step_max_pix: float = DEFAULT_POS_STEP_MAX,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_steps_main: int | None = DEFAULT_MAX_STEPS_MAIN,
    chi2_ftol_rel: float = DEFAULT_CHI2_FTOL_REL,
    step_tol_pix: float = DEFAULT_STEP_TOL_PIX,
    un_freeze_steps: int = DEFAULT_UN_FREEZE_STEPS,
    reg: float = 1e-9,
    nsrc_max: int | None = None,    # fit only buckets with nsrc_class <= this
    joint_nsrc_cap: int = 32,       # islands above this -> giant path
    star_render_split: bool = True,  # PERF #14 mixed star/galaxy render split
    stall_count_mode: str = "proposals",  # #35: "proposals" (legacy) | "changed"
    convergence: str = "legacy",     # dlnp campaign: "legacy" | "dlnp"
    plateau_dlnp_tol: float = 0.5,   # the statistical quantum (2*tol)
    include_giants: bool = True,
    compute_param_errors: bool = False,  # #28a: post-convergence marginal errors
    compute_source_metrics: bool = False,  # #17: per-source rchisq/fracflux/...
    compute_re_profile: bool = False,  # #19a: Re profile-likelihood scan
    return_island_residual: bool = False,  # #34: export per-island coadd residuals
    free_n_override: np.ndarray | None = None,  # #34: force per-source free-n
    log_fn: Optional[Callable[[str], None]] = None,
):
    """Fit all bucketed islands (+ giants), class by class.

    ``joint_nsrc_cap`` (survey note S5: the practical industry limit for
    joint NONLINEAR fits is ~16-25 sources, photutils warns >25; measured
    here: nsrc-64 joint LM at 20 bands exceeds a 24 GB card): islands in
    nsrc buckets ABOVE the cap take the giant path instead (positions and
    shapes kept at the seeds, no island fit, ``island_giant=True``; pass 2
    measures their fluxes).

    ``max_steps_main`` (#25 difficulty re-queue, survey note S5): every
    chunk runs at most this many LM steps in the wide main pass; islands
    that neither converged nor stalled by then are re-queued into
    compact NARROW chunks (``plan.chunk_hard``) for the remaining
    ``max_steps - max_steps_main`` budget, carrying their exact LM
    state. ``None``/``0``/``>= max_steps`` disables the re-queue
    (legacy single-pass behavior).
    """
    import jax.numpy as jnp

    def log(msg):
        if log_fn is not None:
            log_fn(msg)

    n_bands = len(bandnames)
    native = int(psfs[0].H_psf)

    # ---- free-n gate from S1 (production semantics) ----
    # free_n_override (task #34): when given, use it verbatim instead of the
    # S/N gate -- lets the augment path FREEZE detected DET rows' Sersic n
    # (so a DET blob seeded compact can never rail its own n).
    if free_n_override is not None:
        free_n_all = np.asarray(free_n_override, dtype=bool)
    else:
        free_n_all = free_n_gate(source_type, s1_flux, s1_flux_var,
                                 fit_sersic_n, n_fit_min_snr)

    # ---- class planning (bucket -> star/mixed-tier classes + chunks) ----
    plans, capped_giant_ids = plan_island_classes(
        island_set, buckets, source_type, logre, sersic_n, native,
        joint_nsrc_cap=joint_nsrc_cap, nsrc_max=nsrc_max, log_fn=log_fn)

    # ---- difficulty re-queue budgets ----
    if (max_steps_main is None or int(max_steps_main) <= 0
            or int(max_steps_main) >= int(max_steps)):
        budget_main, budget_hard = int(max_steps), 0
    else:
        budget_main = int(max_steps_main)
        budget_hard = int(max_steps) - budget_main

    # Warm the sersic mixture table EAGERLY (rule 2 / sersic_mog TODO).
    if render_backend == "fft" and bool(np.any(source_type != SOURCE_STAR)):
        from . import sersic_mog as _sg
        _sg.get_table_jax()

    # ---- pad image stacks once; upload to device once ----
    occupied_B = max((p.bbox_class for p in plans), default=32)
    PAD = int(occupied_B) + 8
    H, W = data_minus_bg.shape[1], data_minus_bg.shape[2]
    dstack = np.zeros((n_bands, H + 2 * PAD, W + 2 * PAD), dtype=np.float32)
    wstack = np.zeros_like(dstack)
    dstack[:, PAD:PAD + H, PAD:PAD + W] = data_minus_bg
    wstack[:, PAD:PAD + H, PAD:PAD + W] = invvar
    dstack_dev = jnp.asarray(dstack)
    wstack_dev = jnp.asarray(wstack)
    del dstack, wstack

    src_kwargs = dict(
        pos_x=pos_x, pos_y=pos_y, source_type=source_type,
        logre=logre, ee1=ee1, ee2=ee2, sersic_n=sersic_n,
        free_n_all=free_n_all,
        pos_max_shift_pix=float(pos_max_shift_pix), pad=PAD,
        pos_err=pos_err, re_prior_sig=re_prior_sig,
        ell_prior_sig=ell_prior_sig, theta_prior_sig=theta_prior_sig,
        sersic_n_prior_sig=sersic_n_prior_sig,
    )
    out_rows = []
    bucket_stats = []
    island_residuals: dict[int, dict] = {}   # #34 (opt-in)
    psf_tree_cache: dict[int, tuple] = {}
    for plan in plans:
        B, N = int(plan.bbox_class), int(plan.nsrc_class)
        kind, T = plan.kind, int(plan.tier)
        n_isl = plan.n_islands
        t_bucket = time.perf_counter()

        host = class_host_arrays(plan, **src_kwargs)
        if T not in psf_tree_cache:
            psf_tree_cache[T] = build_psf_tree(psfs, T)
        psf_tree, n_terms = psf_tree_cache[T]
        kernels = make_class_kernels(
            plan, n_bands, n_terms, render_backend=render_backend,
            reg=reg, pos_step_max_pix=pos_step_max_pix,
            chi2_ftol_rel=chi2_ftol_rel, step_tol_pix=step_tol_pix,
            un_freeze_steps=un_freeze_steps,
            star_render_split=star_render_split,
            has_priors=bool(host["has_priors"]),
            stall_count_mode=stall_count_mode,
            convergence=convergence,
            plateau_dlnp_tol=plateau_dlnp_tol)

        res_p = np.zeros((n_isl, N, 6), dtype=np.float32)
        res_f = np.zeros((n_isl, n_bands, N), dtype=np.float32)
        res_fe = np.zeros((n_isl, n_bands, N), dtype=np.float32)
        res_chi2 = np.zeros(n_isl, dtype=np.float32)
        res_ngood = np.zeros(n_isl, dtype=np.int64)
        res_conv = np.zeros(n_isl, dtype=bool)
        res_stall = np.zeros(n_isl, dtype=bool)
        res_div = np.zeros(n_isl, dtype=bool)
        # stall-yield certificate (was sharded-only; wired here 2026-07-30
        # so the reference path emits island_stall_dchi2 too).
        res_pdchi2 = np.full(n_isl, np.nan, dtype=np.float32)
        res_nst = np.zeros(n_isl, dtype=np.int32)
        res_perr = np.full((n_isl, N, 13), np.nan, dtype=np.float32)
        res_smetrics = np.full((n_isl, N, 4), np.nan, dtype=np.float32)
        res_swin = np.full((n_isl, n_bands, N), np.nan, dtype=np.float32)
        res_reprof = np.full((n_isl, N, REPROF_DELTAS_LN.size),
                             np.nan, dtype=np.float32)

        def _write(out, rows):
            res_p[rows] = out["p"]
            res_f[rows] = out["fluxes"]
            res_fe[rows] = out["ferr"]
            res_chi2[rows] = out["chi2"]
            if out.get("n_good") is not None:
                res_ngood[rows] = out["n_good"]
            res_conv[rows] = out["conv"]
            res_stall[rows] = out["stall"]
            res_div[rows] = out["diverged"]
            res_pdchi2[rows] = out["stall_dchi2"]
            res_nst[rows] = out["nst"]
            if out["perr"] is not None:
                res_perr[rows] = out["perr"]
            if out.get("smetrics") is not None:
                res_smetrics[rows] = out["smetrics"]
            if out.get("swin") is not None:
                res_swin[rows] = out["swin"]
            if out.get("reprof") is not None:
                res_reprof[rows] = out["reprof"]

        # ---- main pass: wide chunks, low step budget (#25 re-queue) ----
        # T5/#28: same measured width as the sharded path (this reference
        # path runs without compute_flux_marg, so its kernel set -- and
        # therefore its width -- is legitimately its own).
        chunk = int(measure_class_chunk_width(
            kernels, psf_tree, host, n_isl, dstack_dev, wstack_dev, PAD,
            compute_param_errors=compute_param_errors,
            compute_source_metrics=compute_source_metrics,
            compute_re_profile=compute_re_profile,
            want_residual=bool(return_island_residual),
            log_fn=log_fn)["width"])
        if chunk <= 0:
            # Memory-infeasible even one island at a time. The SHARDED path
            # re-routes such a class to the giant path; this
            # reference path has already fixed its giant list, so it tries
            # anyway at width 1 and lets the run fail honestly if it must.
            log(f"bucket bbox={B} nsrc={N} kind={kind} tier={T}: does not "
                f"fit the joint LM one island at a time -- attempting width 1")
        chunk = max(1, chunk)
        n_chunks = int(np.ceil(n_isl / chunk))
        hard_rows_parts: list = []
        hard_state_parts: list = []
        for ci in range(n_chunks):
            lo, hi = ci * chunk, min((ci + 1) * chunk, n_isl)
            rows = np.arange(lo, hi, dtype=np.int64)
            out = run_class_chunk_resilient(
                kernels, psf_tree, host, rows, ladder_width(rows.size, chunk),
                budget_main, dstack_dev, wstack_dev, PAD, log_fn=log_fn,
                compute_param_errors=compute_param_errors,
                compute_source_metrics=compute_source_metrics,
                compute_re_profile=compute_re_profile)
            _write(out, rows)
            if budget_hard > 0:
                uc = ~out["conv"]
                if np.any(uc):
                    hard_rows_parts.append(rows[uc])
                    hard_state_parts.append(
                        {k: out[k][uc] for k in _CARRY_KEYS})

        # ---- difficulty re-queue: compact NARROW hard chunks over the
        # remaining budget, carrying the exact LM state (survey S5) ----
        n_hard = int(sum(r.size for r in hard_rows_parts))
        hard_wall = 0.0
        n_hchunks = 0
        width_h = 0
        if n_hard:
            t_hard = time.perf_counter()
            hrows = np.concatenate(hard_rows_parts)
            hstate = {k: np.concatenate([s[k] for s in hard_state_parts])
                      for k in _CARRY_KEYS}
            # ladder rung (see island_stage._make_hard_tasks): min(chunk//DIV,
            # n_hard) is not one when few islands are re-queued.
            width_h = ladder_width(n_hard, max(1, chunk // HARD_CHUNK_DIV))
            n_hchunks = int(np.ceil(n_hard / width_h))
            for hc in range(n_hchunks):
                sl = slice(hc * width_h, min((hc + 1) * width_h, n_hard))
                out = run_class_chunk_resilient(
                    kernels, psf_tree, host, hrows[sl],
                    ladder_width(int(sl.stop - sl.start), width_h),
                    budget_hard, dstack_dev, wstack_dev, PAD, log_fn=log_fn,
                    init_state={k: hstate[k][sl] for k in _CARRY_KEYS},
                    compute_param_errors=compute_param_errors,
                    compute_source_metrics=compute_source_metrics,
                    compute_re_profile=compute_re_profile)
                _write(out, hrows[sl])
            hard_wall = time.perf_counter() - t_hard

        dt = time.perf_counter() - t_bucket
        bucket_stats.append(dict(
            bbox_class=B, nsrc_class=N, tier=T, kind=kind,
            n_islands=int(n_isl), n_chunks=n_chunks, chunk=chunk,
            wall_s=dt,
            islands_per_s=n_isl / dt if dt > 0 else float("inf"),
            mean_steps=float(res_nst.mean()),
            max_steps_used=int(res_nst.max()),
            n_unconverged=int((~res_conv).sum()),
            n_stalled=int(res_stall.sum()),
            n_requeued=n_hard,
            hard_chunks=n_hchunks, hard_chunk=width_h,
            hard_wall_s=hard_wall, main_wall_s=dt - hard_wall,
        ))
        log(f"bucket bbox={B} nsrc={N} kind={kind}: {n_isl} islands, "
            f"{n_chunks} chunk(s) of {chunk}, {dt:.2f}s "
            f"({n_isl/dt:.0f} isl/s), mean_steps={res_nst.mean():.1f}, "
            f"unconverged={int((~res_conv).sum())}, "
            f"stalled={int(res_stall.sum())}"
            + (f", requeued={n_hard} -> {n_hchunks} hard chunk(s) of "
               f"{width_h} ({hard_wall:.2f}s)" if n_hard else ""))

        # unpack island rows -> source rows (valid member slots only)
        out_rows.append(unpack_class_results(
            plan, host, island_set,
            res_p, res_f, res_fe, res_chi2, res_conv, res_stall, res_nst,
            res_perr=res_perr if compute_param_errors else None,
            res_reprof=res_reprof if compute_re_profile else None,
            reprof_deltas=REPROF_DELTAS_LN,
            res_smetrics=res_smetrics if compute_source_metrics else None,
            res_swin=res_swin if compute_source_metrics else None,
            res_div=res_div,
            res_pdchi2=res_pdchi2,
            res_ngood=res_ngood, n_bands=n_bands))

        # ---- #34 per-island coadded residual export (opt-in) ----
        # One extra 0-step render pass at the converged params, per island,
        # keyed by GLOBAL island id, with the (B,B) window field origin so
        # the detector can map local peaks back to field pixel coords.
        if return_island_residual:
            x0b_p = host["x0b"]
            y0b_p = host["y0b"]
            for ci in range(n_chunks):
                lo, hi = ci * chunk, min((ci + 1) * chunk, n_isl)
                rows = np.arange(lo, hi, dtype=np.int64)
                nc = rows.size
                zi = np.zeros(nc, np.int32)
                rout = run_class_chunk(
                    kernels, psf_tree, host, rows, chunk, 0,
                    dstack_dev, wstack_dev, PAD,
                    init_state=dict(
                        p=res_p[rows], lam=np.zeros(nc, np.float32),
                        chi2=np.zeros(nc, np.float32),
                        consec=zi, nrej=zi.copy(), nst=zi.copy()),
                    want_residual=True)
                for kk, r in enumerate(rows):
                    isl_id = int(plan.isl_ids[r])
                    island_residuals[isl_id] = dict(
                        resid=rout["resid"][kk], resid_w=rout["resid_w"][kk],
                        x0b=float(x0b_p[r]), y0b=float(y0b_p[r]), B=int(B))

    # ---- giants (true giants + S5-capped): seed geometry, no fit (R16) ----
    giant_ids = np.concatenate([
        np.asarray(buckets.giant_ids, dtype=np.int64).ravel(),
        np.asarray(capped_giant_ids, dtype=np.int64),
    ]).astype(np.int64)
    if include_giants and giant_ids.size:
        out_rows.extend(giant_seed_rows(
            giant_ids, island_set, pos_x=pos_x, pos_y=pos_y, logre=logre,
            ee1=ee1, ee2=ee2, sersic_n=sersic_n, n_bands=n_bands))
    elif giant_ids.size:
        log(f"NOTE: {giant_ids.size} giant island(s) skipped "
            f"(include_giants=False)")

    cat = {k: np.concatenate([r[k] for r in out_rows], axis=0)
           for k in out_rows[0]}
    result = IslandFitResult(
        bandnames=list(bandnames),
        source_idx=cat["src"].astype(np.int64),
        pos_x=cat["pos_x"], pos_y=cat["pos_y"],
        logre=cat["logre"], ee1=cat["ee1"], ee2=cat["ee2"],
        sersic_n=cat["sersic_n"],
        fluxes=cat["fluxes"], flux_err=cat["flux_err"],
        chi2=cat["chi2"], chi2_nu=cat["chi2_nu"], n_steps=cat["n_steps"],
        converged=cat["converged"], stalled=cat["stalled"],
        diverged=cat["diverged"],
        stall_dchi2=cat.get("stall_dchi2"),
        free_n=cat["free_n"],
        island_id=cat["island_id"], island_nsrc=cat["island_nsrc"],
        island_giant=cat["island_giant"],
        re_upper_pix=cat["re_upper_pix"],
        pos_x_err=cat["pos_x_err"], pos_y_err=cat["pos_y_err"],
        re_err=cat["re_err"], ell_err=cat["ell_err"],
        re_prof_lo=cat["re_prof_lo"], re_prof_hi=cat["re_prof_hi"],
        theta_err=cat["theta_err"], sersic_n_err=cat["sersic_n_err"],
        pos_x_prior_frac=cat["pos_x_prior_frac"],
        pos_y_prior_frac=cat["pos_y_prior_frac"],
        re_prior_frac=cat["re_prior_frac"],
        ell_prior_frac=cat["ell_prior_frac"],
        theta_prior_frac=cat["theta_prior_frac"],
        sersic_n_prior_frac=cat["sersic_n_prior_frac"],
        rchisq=cat["rchisq"], fracflux=cat["fracflux"],
        fracin=cat["fracin"], fracmasked=cat["fracmasked"],
        flux_win=cat["flux_win"],
        flux_marg_var=cat["flux_marg_var"],
        bucket_stats=bucket_stats,
    )
    if return_island_residual:
        return result, island_residuals
    return result


def island_fit_singletons(**kwargs) -> IslandFitResult:
    """P2-compatible wrapper: fit only the nsrc_class == 1 buckets."""
    return island_fit(nsrc_max=1, include_giants=False, **kwargs)


__all__ = [
    "IslandFitResult",
    "SingletonFitResult",
    "island_fit",
    "island_fit_singletons",
    "build_psf_tree",
    "make_class_kernels",
    "run_class_chunk",
    "run_class_chunk_resilient",
    "measure_class_chunk_width",
    "DEFAULT_MAX_STEPS_MAIN",
]
