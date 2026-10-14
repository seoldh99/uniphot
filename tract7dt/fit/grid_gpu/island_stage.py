"""Field-level island-fit pre-stage for the grid-GPU pipeline.

Wiring for ``compute.gpu.solver.position_solver: island`` (design:
the internal design note ``DESIGN_island_arch.md`` (not distributed); OPT-IN per goal G6):

1. **Pass 1 -- per-grid S1 seed solve.** The pipeline's normal grid
   split + dispatch runs every grid with ``single_stage=True`` and
   ``n_opt_iters=0`` (positions/shapes frozen at the catalog seeds):
   one linear sparse forced-phot solve per grid, giving the seed
   fluxes/variances and the per-grid background planes. The existing
   worker/shm machinery is reused unchanged.
2. **Island stage (this module) -- FIELD level, MULTI-GPU sharded
   (#25, ratified spec 6b).** The parent subtracts each grid's S1
   background from its CORE region and publishes ONE pair of PADDED
   field-level ``data - bg`` / invvar shm stacks (bg FROZEN per design
   D1), decomposes the field's fit sources into flux-aware coupling
   islands (:mod:`.islands`, ``kappa * sigma_sky`` threshold, radii
   from the S1 fluxes) and plans the (bbox, nsrc, kind, tier) solver
   classes host-side (:mod:`.island_plan`, JAX-free). The scheduling
   unit is the CHUNK (a padded vmap batch of one class's islands):
   chunks go into a shared work queue ordered by predicted cost
   descending (LPT), and one pinned spawn worker per CUDA device pulls
   dynamically (program-sticky: a worker prefers chunks of the class
   it just compiled; the persistent JIT cache shares compiled programs
   across workers). Difficulty re-queue (survey S5): every chunk runs
   at most ``max_steps_main`` LM steps; islands that neither converged
   nor stalled continue in compact NARROW "hard" chunks (same kernel
   family) enqueued as soon as their class's main pass drains, carrying
   their exact LM state. Islands are built at the field level -- NOT
   per grid -- so blends straddling grid boundaries are fit jointly;
   this replaces the S2/S3 stages entirely.
3. **Pass 2 -- per-grid S1' final solve.** The per-grid tasks are
   re-issued (again S1-only) at the island-converged positions/shapes/
   Sersic-n via the pipeline's UNCHANGED main dispatch + merge +
   writeback, producing the output fluxes/errors with exact
   cross-island flux consistency. Large-galaxy wings in this native-
   stamp linear solve are handled by the production wing-peeling
   mechanism (``compute.gpu.galaxy.wing_peeling``, default on), which
   now peels at the FITTED shapes.

Notes / limitations (documented behaviour):

- Pixels of grids that own no sources keep bg=0 in the island-stage
  residual (no S1 result exists there); islands never center in such
  grids, at most an island bbox edge reaches in.
- The frozen background is piecewise per grid CORE region (grids share
  the global polynomial normalization, so seams are small).
- Nonlinear-parameter errors (task #28a, default on via
  ``compute.gpu.error_model.param_errors_enabled``): one read-only
  Hessian evaluation at the converged island state yields marginal
  1-sigma errors for position (``x_pix_white_err``/``y_pix_white_err``),
  shape (``Re_err``/``ELL_err``/``THETA_err``) and -- for free-n rows --
  ``sersic_n_err`` (marginal over the island's kept nonlinear params +
  fluxes; background frozen per design D1, unlike the legacy
  joint_varpro conditional flux+bg-marginalized diagonal). Giants and
  frozen/undefined params stay NaN.
- Uses the polynomial PSF model (``compute.gpu.psf.mode`` from_stars).
- Which GPU fits which chunk is a scheduling decision only (chunks are
  independent); results are invariant to the device assignment up to
  the documented chunk-width fp noise class.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from typing import Any

import numpy as np

from ... import run_status
from .island_plan import (
    DEFAULT_CHI2_FTOL_REL, DEFAULT_MAX_STEPS_MAIN, DEFAULT_POS_STEP_MAX,
    DEFAULT_STEP_TOL_PIX, DEFAULT_UN_FREEZE_STEPS, HARD_CHUNK_DIV,
    IslandFitResult, _COST_KIND_FACTOR,
    class_host_arrays, free_n_gate, giant_seed_rows, ladder_width,
    plan_island_classes, unpack_class_results,
    REPROF_DELTAS_LN,
)
from .islands import (
    build_islands, bucket_islands, flux_aware_radii, psf_radial_profile,
)

logger = logging.getLogger("tract7dt.fit.grid_gpu.island_stage")

# effarc2 K6: compiled star-only stamp renderer for the field-wide
# scene-subtract. The python-level cache is keyed on n_terms; jax.jit then
# specialises per ARGUMENT SHAPE, so a run builds one executable per distinct
# (coefficient shape, batch size) it actually calls -- in practice the full
# 2048-source chunk plus the ragged last chunk of each pass, i.e. a handful,
# reused across all bands.
_STAR_STAMPS_JIT_CACHE: dict = {}


def _star_stamps_jit(n_terms: int):
    fn = _STAR_STAMPS_JIT_CACHE.get(int(n_terms))
    if fn is not None:
        return fn
    import types

    import jax
    import jax.numpy as jnp

    from .psf_polynomial import evaluate_psf_at
    from ..optimize.gpu.kernels.lanczos import lanczos_shift_image

    def _body(coefs, x_origin, x_scale, y_origin, y_scale, xs, ys):
        # SimpleNamespace view = the island solver's `_psf_ns` pattern: the
        # polynomial branch of evaluate_psf_at reads only these leaves and
        # takes n_terms from coefs.shape[-1], so the traced program is the
        # same arithmetic as the eager call it replaces.
        psf_ns = types.SimpleNamespace(
            coefs=coefs, x_origin=x_origin, x_scale=x_scale,
            y_origin=y_origin, y_scale=y_scale,
            H_psf=coefs.shape[0], W_psf=coefs.shape[1])

        def _one(x, y):
            raw = evaluate_psf_at(psf_ns, x, y)
            return lanczos_shift_image(raw, x - jnp.round(x), y - jnp.round(y))

        return jax.vmap(_one)(xs, ys)

    fn = jax.jit(_body)
    _STAR_STAMPS_JIT_CACHE[int(n_terms)] = fn
    return fn

# Bucket classes locked by the island-architecture P1.5 field survey
# (UDS + T08285): <= ~21 occupied buckets, slot padding <= 1.24x.
ISLAND_BBOX_CLASSES = (32, 48, 64, 96, 128, 192, 256, 384)
ISLAND_NSRC_CLASSES = (1, 2, 4, 8, 16, 32, 64)


# ---------------------------------------------------------------------------
# Shard-worker side (runs inside pinned spawn workers; initializer =
# runner._worker_init, so JAX_ENABLE_X64=1 + the platform allocator +
# the persistent JIT cache are set before the JAX import).
# ---------------------------------------------------------------------------

_SHARD: dict[str, Any] = {}


def _shard_setup(payload: dict) -> tuple:
    """Once per worker: attach the padded field shm stacks (read-only),
    upload them to THIS worker's device, warm the sersic table."""
    from multiprocessing import shared_memory

    import jax.numpy as jnp

    from .island_solver import (_device_free_bytes_nvml, _gpu_occupants,
                                _is_device_mem_error)

    t0 = time.perf_counter()
    _SHARD.clear()
    attached: list[Any] = []

    def _attach(spec):
        name, shape, dtype = spec
        shm = shared_memory.SharedMemory(name=name)
        attached.append(shm)
        return np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)

    dpad = _attach(payload["data_spec"])       # (n_bands, Hp, Wp), PADDED
    wpad = _attach(payload["invvar_spec"])

    # ---- field-stack budget pre-flight + attribution (2026-08-30) --------
    # These two uploads are the ONE fixed device requirement of the island
    # stage: an island is cut out of the padded field in-kernel, so there is
    # no smaller version of them. Everything else here is already adaptive --
    # `measure_class_chunk_width` sizes the chunk against `_device_free_bytes`
    # and `run_class_chunk_resilient` halves the width on a memory failure --
    # which is exactly why a contended card kills the run HERE, before any of
    # that can run (CANDIDATES_v1 SS7 N1; deaths 2026-08-29 02:14 and
    # 2026-08-30 03:05, 41 and 43 wasted WALL-minutes, neither log naming the
    # card nor the occupant -- and neither window is covered by the gpumon
    # sampler either, so both deaths remain UNATTRIBUTED BY MEASUREMENT. What
    # the tracebacks do force is an unmeasured occupant holding >=17 GiB: the
    # first stack alone is 5,553 MiB and our own footprint at that instant was
    # 1.7 GiB. The launcher-side gate is measured-insufficient -- run_e9's
    # `wait_for_room` (18.33 GiB/card) PASSED at 02:24:10 and the run died at
    # 03:05 anyway -- because it samples once, before the run.)
    #
    # No constant is introduced. The requirement is the exact byte count of
    # the two arrays the PARENT already sized when it created the shm
    # segments; the budget is this card's free memory from the same meter the
    # width ladder uses. No reservation is estimated either: this worker's own
    # CUDA context already exists when `_shard_setup` runs (`_worker_init`
    # warms it), so the driver's free number has already subtracted it.
    phys = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    need = int(dpad.nbytes) + int(wpad.nbytes)

    def _fail(reason: str, free: int, exc: "BaseException | None" = None):
        me = os.getpid()
        who = ", ".join(
            f"pid {p} ({u}{', this worker' if p == me else ''}) "
            f"{b / 2 ** 30:.2f} GiB"
            for p, b, u in _gpu_occupants(phys)) or "none reported"
        raise RuntimeError(
            f"island shard worker on PHYSICAL GPU {phys}: {reason}. "
            f"Field stacks need {need / 2 ** 30:.2f} GiB ({need} B = data "
            f"{int(dpad.nbytes)} + invvar {int(wpad.nbytes)}); card reports "
            f"{free / 2 ** 30:.2f} GiB free ({int(free)} B); compute "
            f"processes on GPU {phys}: {who}") from exc

    free0 = _device_free_bytes_nvml()
    # `0 <` matters: the meter returns 0 when it cannot read the card, and a
    # missing meter must NEVER turn a run that would have worked into a
    # failure. A positive reading below `need` cannot succeed -- both stacks
    # must be resident at once -- so failing here only replaces the same death
    # with a diagnosable one, and does it before the first 5.42 GiB is
    # stranded on the card by an upload whose partner cannot follow.
    if 0 < free0 < need:
        _fail("not enough free device memory for the field stacks", free0)
    try:
        dstack_dev = jnp.asarray(dpad)
        wstack_dev = jnp.asarray(wpad)
        dstack_dev.block_until_ready()
        wstack_dev.block_until_ready()
    except Exception as exc:                                   # noqa: BLE001
        # Allocator granularity, fragmentation, or an occupant that arrived
        # between the read and the upload. Same message, measured after.
        if not _is_device_mem_error(exc):
            raise
        _fail(f"{type(exc).__name__} uploading the field stacks",
              _device_free_bytes_nvml(), exc)
    if payload["warm_sersic_table"]:
        # Engineering rule 2 (island_solver docstring): the lazy table
        # cache must never be populated inside a jit trace.
        from . import sersic_mog as _sg
        _sg.get_table_jax()
    _SHARD.update(
        payload=payload, attached=attached, dpad=dpad, wpad=wpad,
        dstack_dev=dstack_dev, wstack_dev=wstack_dev,
        class_cache={}, psf_tree_cache={},
    )
    return (os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
            time.perf_counter() - t0)


def _shard_class_setup(ci: int) -> tuple:
    """Lazy per-class prep in this worker: host arrays + kernels + PSF
    tree (cached; the kernel cache key is the class shape, so re-pulls
    of the same class reuse the compiled programs -- program-sticky)."""
    cache = _SHARD["class_cache"]
    got = cache.get(ci)
    if got is not None:
        return got
    from .island_solver import build_psf_tree, make_class_kernels

    payload = _SHARD["payload"]
    plan = payload["plans"][ci]
    host = class_host_arrays(plan, **payload["src_kwargs"])
    T = int(plan.tier)
    ptc = _SHARD["psf_tree_cache"]
    if T not in ptc:
        ptc[T] = build_psf_tree(payload["psfs"], T)
    psf_tree, n_terms = ptc[T]
    kernels = make_class_kernels(
        plan, len(payload["bandnames"]), n_terms,
        has_priors=bool(host["has_priors"]), **payload["solver_opts"])
    cache[ci] = (plan, host, kernels, psf_tree)
    return cache[ci]


def _shard_note_width(ci: int, width: int) -> None:
    """Record that this worker has BUILT the class-``ci`` kernel set at
    ``width`` (effarc2 K12 bookkeeping)."""
    _SHARD.setdefault("built_widths", {}).setdefault(int(ci), set()).add(
        int(width))


def _shard_note_chunk(ci: int, width: int, steps: int, wall_s: float) -> None:
    """Record one observed chunk cost for class ``ci`` (effarc2 R2)."""
    if width <= 0 or wall_s <= 0:
        return
    _SHARD.setdefault("chunk_obs", {}).setdefault(int(ci), []).append(
        (int(width), max(1, int(steps)), float(wall_s)))


def _shard_cost_model(ci: int):
    """(per-lane-step seconds, fresh-rung compile seconds) MEASURED for this
    class on this worker, or None while there is not enough evidence.

    effarc2 R2 (2026-08-28). Both numbers come from chunks this worker has
    already run, so no constant is introduced:

    * ``rate`` = the MINIMUM of ``wall / (width x steps)`` over the
      observations. A chunk that had to compile its width carries that cost in
      its wall, so the minimum is the compile-free per-lane-step cost.
    * ``compile_s`` = the LARGEST observed excess of a chunk's wall over
      ``rate x width x steps``. That excess is what a chunk pays the first time
      its shape is built.

    Two observations are required: with one, ``rate`` and the excess are the
    same measurement and the model would be vacuous.
    """
    obs = _SHARD.get("chunk_obs", {}).get(int(ci)) or []
    if len(obs) < 2:
        return None
    rate = min(w_s / (w * st) for w, st, w_s in obs)
    if not (rate > 0):
        return None
    compile_s = max(0.0, max(w_s - rate * w * st for w, st, w_s in obs))
    return rate, compile_s


def _shard_reuse_width(ci: int, n_rows: int, width: int,
                       is_phase_a: bool, max_steps: int = 0) -> int:
    """Widen a chunk to a rung this worker has ALREADY built, when the
    worker's own measurements say padding is cheaper than compiling the
    narrow rung.

    effarc2 K12 (2026-08-26) + R2 (2026-08-28). The ladder design assumed a
    narrow tail rung is free because "the cache already has it" -- true on a
    warm cache, false on the run that pays for it. Measured on the sparse cold
    anchor: a 1-island tail at width 1 took 20.2 s and a 45-island tail at
    width 64 took 17.5 s, while the SAME class's already-built 128-wide chunk
    took 0.6 s. Those rungs were pure XLA compile.

    But padding is not free either, and whether it wins depends on the class:
    on the crowded anchor an unconditional rule ran class (96,32)'s 2-island
    re-queue chunks at width 256 for 80.5 s and 71.6 s where width 2 costs
    32.4 s, while class (32,1) ran its 1,024-island re-queue chunks at width
    4096 in 1.1 s against 58.7 s of fresh-rung compiles. So the decision is
    made from :func:`_shard_cost_model` -- the per-lane-step cost and the
    fresh-rung compile cost, both measured on this class by this worker:

        pad   = rate x w_new     x max_steps        (run the wider, built rung)
        tight = compile + rate x width x max_steps  (build the narrow rung)

    ``max_steps`` matters because it is the phase-B multiplier: a re-queue
    chunk carries ``budget_hard`` (22) against a main-pass mean of 1.2-4.5.

    Until two chunks of the class have been observed the model is vacuous. The
    fallback is the pre-R2 behaviour, which is safe for the reason it was
    chosen: a widened PHASE-A tail runs the same ``budget_main`` ceiling as
    every other phase-A chunk, so its cost is bounded above by one ordinary
    chunk of that class -- a cost the run already pays n_chunks times. Phase B
    has no such bound, so with no evidence it is left alone.

    Padding itself is inert by construction: ``_build_chunk_inputs`` pads with
    mmask/keep/gvalid False, ``run_class_chunk`` seeds padded lanes as
    CONVERGED (they never take an LM step and cannot block the all-converged
    early exit), and every output is sliced back to ``n_rows``. Batch
    composition is not fp-neutral though (T3 C2/C3), so this is an fp_noise
    change, not a bit-identical one.
    """
    if is_phase_a and int(n_rows) >= int(width):
        # A FULL phase-A chunk already runs at the class's measured width,
        # which the probe built: no compile to avoid, so extra lanes are pure
        # waste. NOT applied to phase B -- a full phase-B chunk sits on a
        # narrow re-queue rung that nothing has compiled yet.
        return int(width)
    built = _SHARD.get("built_widths", {}).get(int(ci))
    if not built:
        return int(width)
    fits = [w for w in built if w >= int(n_rows) and w > int(width)]
    if not fits:
        return int(width)
    w_new = int(min(fits))
    model = _shard_cost_model(int(ci))
    if model is not None and int(max_steps) > 0:
        rate, compile_s = model
        pad_cost = rate * w_new * int(max_steps)
        tight_cost = compile_s + rate * int(width) * int(max_steps)
        return w_new if pad_cost <= tight_cost else int(width)
    return w_new if is_phase_a else int(width)


def _shard_measure_width(ci: int, want_residual: bool = False,
                         cached: dict | None = None,
                         min_chunks: int = 1,
                         cost_table: dict | None = None,
                         class_ladder: bool = False) -> dict:
    """T5/#28: this worker measures the chunk width for class ``ci`` on ITS
    OWN device and returns the verdict to the parent scheduler.

    It has to happen here, not in the (JAX-free) parent: the budget is this
    card's free memory with the field stacks already resident, and the cost
    is what XLA reports for the compiled kernels -- neither exists outside a
    worker holding the device. Doing it here also warms this worker's kernel
    cache for the class it is about to be handed.

    ``cached`` (width bundle #3): a prior verdict for THIS class key whose
    parent-level key (engine version, devices, bands, kernel flags, solver
    opts) already matched. It is adopted -- probes and the verify compile
    skipped -- only if the ladder limit it was measured under still holds
    AND its measured cost still fits this card's CURRENT free memory
    (downward-safe: shrunken memory re-measures). The verify compile lost
    by skipping is the production executable the measuring pass already
    left in the persistent JIT cache.
    """
    from .island_solver import (measure_class_chunk_width,
                                effective_rung_limit, _LADDER_MAX_WIDTH,
                                _device_free_bytes)

    plan, host, kernels, psf_tree = _shard_class_setup(ci)
    payload = _SHARD["payload"]
    _rung = effective_rung_limit(int(plan.n_islands), _LADDER_MAX_WIDTH,
                                 int(min_chunks))
    if cached is not None and int(cached.get("width", 0)) > 0:
        try:
            _free = _device_free_bytes()
        except Exception:
            _free = None
        if (int(cached.get("rung_limit", -1)) == _rung
                and _free is not None
                and float(cached["cost_bytes"]) <= float(_free)):
            res = dict(cached)
            res.update(probes=[], reused=True, wall_s=0.0,
                       budget_bytes=float(_free))
            res["ci"] = int(ci)
            res["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            return res
    res = measure_class_chunk_width(
        kernels, psf_tree, host, plan.n_islands,
        _SHARD["dstack_dev"], _SHARD["wstack_dev"], int(payload["pad"]),
        # R1 (effarc2, 2026-08-28): measure with the MAXIMAL optional-kernel
        # set regardless of what this pass actually runs. These three flags
        # are part of `_shape_key`, so a pass that measured with them off
        # writes a cache entry the other pass cannot key into -- which is how
        # dev2 made pass B re-probe every class. Pinning them True makes the
        # key a constant, so the phantom pass's two children SHARE one width
        # verdict, and pass A is then free to skip actually RUNNING them (K8).
        # Measuring high is also the safe direction: the adopted width fits
        # the largest kernel set, so a pass running fewer kernels cannot OOM.
        compute_param_errors=True,
        compute_source_metrics=True,
        compute_flux_marg=True,
        compute_re_profile=bool(payload.get("compute_re_profile", False)),
        want_residual=bool(want_residual),
        min_chunks=int(min_chunks),
        cost_table=cost_table,
        class_ladder=bool(class_ladder),
        log_fn=lambda m: logger.info("island stage[shard]: %s", m))
    res["ci"] = int(ci)
    res["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    res["rung_limit"] = _rung
    res["reused"] = False
    if int(res.get("width", 0)) > 0 and res.get("verified"):
        # the verify pass built the production kernels at this width here
        _shard_note_width(ci, int(res["width"]))
    return res


def _shard_run_chunk(
    ci: int, rows: np.ndarray, width: int, max_steps: int,
    init_state: dict | None,
    want_residual: bool = False,
    compute_param_errors: bool | None = None,
) -> dict:
    """Fit one chunk (the scheduling unit) on this worker's device.

    ``want_residual`` / ``compute_param_errors`` (both default to the
    main-fit behaviour: no residual, payload's param-errors flag) are the
    #34 residual-export levers. A 0-step ``want_residual=True`` call at the
    converged params returns the per-island coadded model-subtracted
    residual (``resid``/``resid_w``) for the un-attributed detector."""
    from .island_solver import run_class_chunk_resilient

    _plan, host, kernels, psf_tree = _shard_class_setup(ci)
    # phase A <=> no carried LM state (`_seed_class_tasks` passes init_state
    # None; `_make_hard_tasks` always carries the re-queue state).
    width = _shard_reuse_width(ci, int(np.asarray(rows).size), int(width),
                               init_state is None, int(max_steps))
    cpe = (bool(_SHARD["payload"].get("compute_param_errors", False))
           if compute_param_errors is None else bool(compute_param_errors))
    crp = bool(_SHARD["payload"].get("compute_re_profile", False))
    csm = bool(_SHARD["payload"].get("compute_source_metrics", False))
    cfm = bool(_SHARD["payload"].get("compute_flux_marg", False))
    out = run_class_chunk_resilient(
        kernels, psf_tree, host, rows, width, max_steps,
        _SHARD["dstack_dev"], _SHARD["wstack_dev"],
        _SHARD["payload"]["pad"], init_state=init_state,
        log_fn=lambda m: logger.warning("island stage[shard]: %s", m),
        compute_param_errors=cpe, want_residual=bool(want_residual),
        compute_re_profile=crp,
        compute_source_metrics=csm, compute_flux_marg=cfm)
    out["gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    # Record the width that ACTUALLY ran. `run_class_chunk_resilient` halves on
    # a device-memory error, so the requested width may never have been built
    # (and a width that failed here must not become a widening target for a
    # later chunk on this worker).
    _ran = out.pop("widths_ran", None) or [int(width)]
    out["width_used"] = int(max(_ran))
    for _w in _ran:
        _shard_note_width(ci, int(_w))
    if not out.pop("halved", False) and len(_ran) == 1:
        # Only an UN-HALVED chunk is a clean cost observation. Both conditions
        # are load-bearing and neither implies the other: `widths_ran` is a
        # deduplicated set, so a uniform OOM halving also has length 1 (and
        # carries the SUM of both leaves' wall time -- a ~2x error in the rate
        # the model derives); the length check still rejects a NON-uniform or
        # nested halving, where the flag alone would be enough but the width
        # to attribute the cost to is ambiguous anyway.
        _shard_note_chunk(ci, int(_ran[0]), int(out.get("steps_run", 1)),
                          float(out.get("wall_s", 0.0)))
    return out


def _shard_close() -> bool:
    """Detach the shm stacks cleanly before the pool shuts down."""
    for shm in _SHARD.get("attached", []):
        try:
            shm.close()
        except Exception:  # noqa: BLE001
            pass
    _SHARD.clear()
    return True


# ---------------------------------------------------------------------------
# Parent-side scheduler
# ---------------------------------------------------------------------------


def _report_shard_setup(devices: list[int], setup_futs: list) -> None:
    """Await EVERY shard-setup future, report per device, then re-raise.

    The pre-2026-08-30 form was
    ``for d, fut in zip(devices, setup_futs): seen, dt = fut.result()``,
    which raises on the FIRST failing future and silently discards its
    siblings' outcomes. That is why the 2026-08-29 02:14 and 2026-08-30 03:05
    crowded-gate deaths preserved exactly one traceback and were written up as
    "all three cards failed" (corrected in development ledger SS23-30: only
    the ``devices[0]`` worker is proven to have failed). The distinction is
    the whole diagnosis -- one card is ordinary contention, three would be an
    engine bug -- and it cost two 40-minute runs to not have it.

    Waiting for all of them also keeps the caller's ``finally`` teardown from
    racing a sibling that is still mid-upload. On a run where every worker
    succeeds this is exactly the old loop: same log lines, same order, and
    the wait returns when the last future is done, which is when the old
    loop's last ``result()`` would have returned.
    """
    futures_wait(setup_futs)
    failed: list[tuple[int, BaseException]] = []
    for d, fut in zip(devices, setup_futs):
        exc = fut.exception()
        if exc is not None:
            failed.append((int(d), exc))
            logger.error("grid_gpu[island]: shard worker GPU %d setup "
                         "FAILED: %s: %s", int(d), type(exc).__name__, exc)
            continue
        seen, dt = fut.result()
        logger.info("grid_gpu[island]: shard worker GPU %s ready "
                    "(field stacks on device in %.1fs)", seen, dt)
    if failed:
        bad = {d for d, _ in failed}
        logger.error(
            "grid_gpu[island]: shard setup failed on %d of %d card(s): "
            "failed GPU(s) %s, ready GPU(s) %s -- re-raising the first "
            "failure", len(failed), len(devices), sorted(bad),
            [int(d) for d in devices if int(d) not in bad])
        raise failed[0][1]


def _run_island_fit_sharded(
    *,
    devices: list[int],
    ctx,
    mem_fraction: float,
    jit_cache_dir,
    worker_log_file,
    payload: dict,
    plans: list,
    island_set,
    giant_ids: np.ndarray,
    budget_main: int,
    budget_hard: int,
    n_bands: int,
    export_residual: bool = False,
    width_cache_path=None,
) -> tuple[IslandFitResult, dict, dict | None]:
    """LPT + program-sticky dynamic scheduling of island chunks over one
    pinned worker per CUDA device (ratified sharding spec, design 6b).

    ``export_residual`` (#34 residual detect path): after the main+hard
    scheduling drains, run ONE extra 0-step ``want_residual`` chunk per
    class at the converged params (workers still alive, field stacks still
    on device) to export the per-island coadded model-subtracted residual,
    keyed by GLOBAL island id -- exactly the reference
    ``island_fit(return_island_residual=True)`` export, sharded.

    Returns ``(IslandFitResult, shard_stats, island_residuals)``;
    ``island_residuals`` is ``None`` unless ``export_residual``."""
    src_kwargs = payload["src_kwargs"]

    pending: list[dict] = []
    class_track: dict[int, dict] = {}
    extra_giant_ids: list[int] = []

    def _seed_class_tasks(plan, chunk: int, per_island_cost: float):
        """Phase-A chunk tasks for one class at its MEASURED width."""
        n_isl = plan.n_islands
        N = int(plan.nsrc_class)
        chunk = int(max(1, min(chunk, n_isl)))
        n_chunks = int(np.ceil(n_isl / chunk))
        exp_steps = (budget_main if plan.kind == "mixed"
                     else min(4, budget_main))
        for ci in range(n_chunks):
            lo, hi = ci * chunk, min((ci + 1) * chunk, n_isl)
            # T5/#28: the TAIL chunk runs at the smallest LADDER width that
            # holds its rows instead of padding out to the full chunk --
            # the ladder is what makes that free (the narrow width is a
            # rung, so it is a shape the cache already has).
            pending.append(dict(
                type="chunk", phase="A", ci=int(plan.class_idx),
                rows=np.arange(lo, hi, dtype=np.int64),
                width=ladder_width(hi - lo, chunk),
                max_steps=budget_main, init_state=None,
                cost=float(hi - lo) * per_island_cost * exp_steps,
            ))
        class_track[int(plan.class_idx)] = dict(
            outstanding=n_chunks, hard_outstanding=0,
            hard_rows=[], hard_state=[],
            res_p=np.zeros((n_isl, N, 6), np.float32),
            res_f=np.zeros((n_isl, n_bands, N), np.float32),
            res_fe=np.zeros((n_isl, n_bands, N), np.float32),
            res_chi2=np.zeros(n_isl, np.float32),
            res_ngood=np.zeros(n_isl, np.int64),
            res_conv=np.zeros(n_isl, bool),
            res_stall=np.zeros(n_isl, bool),
            res_div=np.zeros(n_isl, bool),
            res_pdchi2=np.full(n_isl, np.nan, np.float32),
            res_railed=np.zeros(n_isl, bool),
            res_nst=np.zeros(n_isl, np.int32),
            res_perr=np.full((n_isl, N, 13), np.nan, np.float32),
            res_smetrics=np.full((n_isl, N, 4), np.nan, np.float32),
            res_swin=np.full((n_isl, n_bands, N), np.nan, np.float32),
            res_reprof=np.full((n_isl, N, REPROF_DELTAS_LN.size),
                               np.nan, np.float32),
            # D2: per-band SHAPE-marginalized flux-error INCREMENT variance
            # (n_isl, n_bands, N); 0 for star rows. NaN default so off-island
            # / never-filled slots are honest.
            res_femarg=np.full((n_isl, n_bands, N), np.nan, np.float32),
            gpu_s=0.0, main_gpu_s=0.0, hard_gpu_s=0.0,
            n_hard=0, hard_chunks=0, hard_chunk=0,
            n_chunks=n_chunks, chunk=chunk,
            per_island_cost=float(per_island_cost),
        )

    def _in_submit_order(done, book):
        """Iterate finished futures in SUBMISSION order, not set order.

        `concurrent.futures.wait` returns a set; iterating it is Python
        object-hash order, which varies between processes. `book` is the
        running-task dict, which preserves insertion (= submission) order,
        so indexing into it gives a stable, run-independent sequence.
        """
        pos = {f: i for i, f in enumerate(book)}
        return sorted(done, key=lambda f: pos.get(f, -1))

    exp_hard_steps = max(1, budget_hard // 2)

    def _make_hard_tasks(plan, tr) -> list[dict]:
        hrows = np.concatenate(tr["hard_rows"])
        hstate = {k: np.concatenate([s[k] for s in tr["hard_state"]])
                  for k in tr["hard_state"][0]}
        tr["hard_rows"], tr["hard_state"] = [], []   # release
        # The blocks above were appended in phase-A COMPLETION order, and
        # the chunks below are cut by POSITION, so without this the phase-B
        # chunk membership (and each island's vmap lane) is a function of
        # worker timing. Chunk geometry demonstrably reaches the numbers --
        # development ledger SS23-18 measured 6.50 mmag from changing the
        # width alone -- so this is a value channel, not just bookkeeping.
        # Sorting by row index makes the partition a pure function of the
        # class's own island set. Stable sort: each phase-A chunk
        # contributes an already-ascending block, so this is a merge.
        _perm = np.argsort(hrows, kind="stable")
        hrows = hrows[_perm]
        hstate = {k: v[_perm] for k, v in hstate.items()}
        n_hard = int(hrows.size)
        # Narrower re-queue width, and it must be a LADDER RUNG too.
        # `min(chunk//DIV, n_hard)` is not: when fewer islands are re-queued
        # than the narrow width, it becomes n_hard itself -- an arbitrary
        # integer that also varies run to run (how many islands failed to
        # converge is data- and fp-dependent). Measured 2026-08-03: phase-A
        # widths were 100% powers of two while phase-B leaked 3/5/6/156/1016,
        # which costs cross-field kernel reuse and adds run-dependent shapes.
        width_h = ladder_width(n_hard, max(1, tr["chunk"] // HARD_CHUNK_DIV))
        n_h = int(np.ceil(n_hard / width_h))
        tr["n_hard"] = n_hard
        tr["hard_chunks"] = n_h
        tr["hard_chunk"] = width_h
        tr["hard_outstanding"] = n_h
        tasks = []
        for hc in range(n_h):
            sl = slice(hc * width_h, min((hc + 1) * width_h, n_hard))
            tasks.append(dict(
                type="chunk", phase="B", ci=int(plan.class_idx),
                rows=hrows[sl], max_steps=budget_hard,
                width=ladder_width(int(sl.stop - sl.start), width_h),
                init_state={k: v[sl] for k, v in hstate.items()},
                cost=(float(sl.stop - sl.start) * tr["per_island_cost"]
                      * exp_hard_steps),
            ))
        return tasks

    def _class_done_log(plan, tr):
        n_isl = plan.n_islands
        dt = tr["gpu_s"]
        logger.info(
            "island stage: bucket bbox=%d nsrc=%d kind=%s: %d islands, "
            "%d chunk(s) of %d, %.2fs GPU (%.0f isl/s), mean_steps=%.1f, "
            "unconverged=%d, stalled=%d%s",
            plan.bbox_class, plan.nsrc_class, plan.kind, n_isl,
            tr["n_chunks"], tr["chunk"], dt,
            n_isl / dt if dt > 0 else float("inf"),
            float(tr["res_nst"].mean()),
            int((~tr["res_conv"]).sum()), int(tr["res_stall"].sum()),
            (f", requeued={tr['n_hard']} -> {tr['hard_chunks']} hard "
             f"chunk(s) of {tr['hard_chunk']} ({tr['hard_gpu_s']:.2f}s GPU)"
             if tr["n_hard"] else ""))

    # ---- spawn one pinned worker per device; broadcast the payload ----
    from .runner import _worker_init

    n_slots = len(devices)
    executors: list[ProcessPoolExecutor] = []
    t_stage0 = time.perf_counter()
    busy: dict[str, float] = {}
    per_task_log: list[dict] = []
    try:
        for d in devices:
            executors.append(ProcessPoolExecutor(
                max_workers=1, mp_context=ctx, initializer=_worker_init,
                initargs=(d, mem_fraction, jit_cache_dir, worker_log_file,
                          "platform"),
            ))
        setup_futs = [ex.submit(_shard_setup, payload) for ex in executors]
        _report_shard_setup(devices, setup_futs)

        # ---- T5/#28: MEASURE each class's chunk width on the cards -------
        # Has to come after _shard_setup (the budget is free memory WITH the
        # field stacks resident) and before the task list (the width is the
        # scheduling unit). Spread over the workers; the persistent JIT
        # cache means the second worker to meet a shape pays nothing.
        #
        # Width bundle #3 (development ledger SS15): the phantom pass B used
        # to re-measure the exact ladder pass A had just measured (widths
        # observed identical on every class, sparse+crowded). A pass now
        # persists its verdicts to {work_dir}/island_chunk_widths.json; a
        # later pass in the same work_dir adopts a class's verdict only when
        # the parent-level key below matches AND the worker-side check holds
        # (same ladder limit, cached cost fits the card's CURRENT free
        # memory). Any mismatch re-measures, per class. Disabled under
        # TRACT7DT_ISLAND_CHUNK (a forced width would poison the file).
        import json as _json
        from ... import __version__ as _t7ver
        # I3-lite (2026-08-24, development ledger SS21-8/9): `pad` is NOT part
        # of the width key any more. PAD = max(occupied bbox class)+8 is a
        # TILE property, but the probe measures only the solver kernels,
        # whose shapes are (width, n_bands, B, B)/(width, 6N) -- PAD-free
        # (island_solver._kernel_calls; the PAD-shaped gather runs outside
        # the measured list and costs 0 floor-seconds). Keying the table on
        # pad fragmented cross-tile reuse for no physical reason (measured:
        # T12381, pad 104 vs 72, hit 0/20 and re-probed everything; costs
        # identical across the flip). Safety is unchanged and structural:
        # adoption re-validates the rung limit and that the cached cost
        # fits THIS card's current free memory, `_verify` still compiles
        # the chosen width for real, and the OOM-halving retry stands -- a
        # wrong entry can cost performance, never science.
        _shape_key = {
            "n_bands": int(n_bands),
            # The key records the PROBE SET, not the run's flags. The three
            # below are measured unconditionally (see `_shard_measure_width`),
            # so they are constants here -- that is exactly what makes the
            # verdict shareable between the residual-phantom pass's two
            # children. Written as literals rather than dropped so an old
            # cache file (whose key carried the run's flags) simply fails to
            # match and is re-measured instead of being adopted wrongly.
            "flags": {
                "param_errors": True,
                "source_metrics": True,
                "flux_marg": True,
                "re_profile": bool(
                    payload.get("compute_re_profile", False)),
                "residual": bool(export_residual),
            },
            "solver_opts": {str(k): str(v) for k, v in
                            sorted(payload["solver_opts"].items())},
        }
        cache_key = None
        cached_classes: dict = {}

        def _wckey(pl):
            return (f"{int(pl.bbox_class)},{int(pl.nsrc_class)},"
                    f"{pl.kind},t{int(pl.tier)}")

        if (width_cache_path is not None
                and not os.environ.get("TRACT7DT_ISLAND_CHUNK")):
            from .island_solver import kernel_fingerprint as _kfp
            cache_key = dict(_shape_key, engine=str(_t7ver),
                             kernels=_kfp(),
                             devices=[int(d) for d in devices])
            try:
                with open(width_cache_path) as _fh:
                    _cc = _json.load(_fh)
                if _cc.get("key") == cache_key:
                    cached_classes = _cc.get("classes", {})
            except FileNotFoundError:
                pass
            except Exception as _e:
                logger.warning(
                    "island width cache unreadable (%s) -- measuring fresh",
                    _e)
        # ---- width bundle #2 (global): cross-tile XLA cost table ---------
        # (development ledger SS17) The probe compiles are pure sunk cost
        # (no-autotune executables production never reuses), and the cost
        # they measure is a deterministic function of the kernel SHAPES
        # (verified identical across 5 runs / code generations at log
        # precision). So probe verdicts are cached per (engine, jax,
        # device model, shape key, class, width) INSIDE the persistent JIT
        # cache dir -- same invalidation regime, shared across tiles.
        # Safety is structural: _verify still compiles the chosen width
        # for real and the runtime OOM-halving retry stands; a wrong entry
        # can cost performance, never science. Misses probe as before and
        # are merged back; a fresh probe disagreeing with the table WARNs
        # and the fresh value wins.
        cost_table_path = None
        cost_meta = None
        cost_entries: dict = {}
        class_costs: dict[int, dict] = {}
        _sig = _json.dumps(_shape_key, sort_keys=True)
        if (jit_cache_dir is not None and str(jit_cache_dir) != "none"
                and not os.environ.get("TRACT7DT_ISLAND_CHUNK")):
            _cdir = (os.path.expanduser(
                         "~/.cache/tract7dt/jax_jit_cache_grid")
                     if str(jit_cache_dir) == "auto" else str(jit_cache_dir))
            try:
                from importlib.metadata import version as _pkgver
                import subprocess as _sp
                _gq = _sp.run(
                    ["nvidia-smi", "--query-gpu=name",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10).stdout
                from .island_solver import kernel_fingerprint as _kfp2
                cost_meta = {
                    "engine": str(_t7ver),
                    # effarc2 R3: `engine` is the INSTALLED package version and
                    # does not move when an uncommitted working tree is
                    # shipped by copying files; the kernel-source hash does.
                    "kernels": _kfp2(),
                    "jax": _pkgver("jax"), "jaxlib": _pkgver("jaxlib"),
                    "devices": sorted({x.strip() for x in _gq.splitlines()
                                       if x.strip()}),
                }
                cost_table_path = os.path.join(
                    _cdir, "width_cost_table.json")
            except Exception as _e:
                logger.warning(
                    "width cost table disabled (meta unavailable: %s)", _e)
            if cost_table_path is not None:
                try:
                    with open(cost_table_path) as _fh:
                        _ct = _json.load(_fh)
                    if _ct.get("meta") == cost_meta:
                        cost_entries = _ct.get("entries", {})
                    else:
                        logger.info(
                            "width cost table: meta changed (engine/jax/"
                            "device) -- starting fresh")
                except FileNotFoundError:
                    pass
                except Exception as _e:
                    logger.warning(
                        "width cost table unreadable (%s) -- probing "
                        "fresh", _e)
                for pl in plans:
                    _ent = cost_entries.get(_sig + "|" + _wckey(pl))
                    if _ent:
                        class_costs[int(pl.class_idx)] = {
                            int(w): (int(v[0]), str(v[1]))
                            for w, v in _ent.items()}
        t_meas = time.perf_counter()
        # NOTE (width bundle #1, 2026-08-14): a topology-derived rung cap
        # (>= n_devices chunks per big class) was implemented and MEASURED
        # NEGATIVE on the sparse anchor -- width IS the vmap batch, so
        # thinner chunks halved GPU throughput (fit stage 66.4 -> 142.2 s,
        # ideal makespan 26.2 -> 51.8 s) and broke the pass-A/B width
        # identity near the cap boundary. Reverted to min_chunks=1 (the
        # historical limit); see development ledger SS15. A future cap must
        # weigh the measured isl/s rate against balance, not chunk counts.
        meas_futs = {
            executors[i % n_slots].submit(
                _shard_measure_width, int(pl.class_idx),
                bool(export_residual), cached_classes.get(_wckey(pl)),
                cost_table=class_costs.get(int(pl.class_idx)),
                # T6/I1 'probe the class' (fill the cross-tile table to the
                # class ceiling) was built and MEASURED NEGATIVE 2026-08-23
                # (development ledger SS21-6): the table key carries PAD =
                # max occupied bbox + 8, so tiles with a different largest
                # class never hit, and where they do hit the probe phase is
                # dominated by the production (verify) compiles of new main
                # widths, not by the no-autotune probes; first encounter
                # cost +37..+65 s per tile. Kept available (class_ladder in
                # island_solver.measure_class_chunk_width) for a future
                # tile-independent PAD; OFF here.
                class_ladder=False
            ): int(pl.class_idx) for i, pl in enumerate(plans)}
        measured: dict[int, dict] = {}
        for fut in list(meas_futs):
            m = fut.result()
            measured[int(m["ci"])] = m
        n_probes = sum(len(m["probes"]) for m in measured.values())
        n_reused = sum(1 for m in measured.values() if m.get("reused"))
        if width_cache_path is not None and cache_key is not None:
            try:
                _classes_out = {}
                for pl in plans:
                    m = measured.get(int(pl.class_idx))
                    if m is None or int(m.get("width", 0)) <= 0:
                        continue  # giant downgrades always re-measure fresh
                    _classes_out[_wckey(pl)] = {
                        "width": int(m["width"]),
                        "cost_bytes": float(m["cost_bytes"]),
                        "per_island_bytes": float(m["per_island_bytes"]),
                        "rung_limit": int(m.get("rung_limit", 0)),
                        "dominant": str(m.get("dominant", "")),
                    }
                _tmp = str(width_cache_path) + ".tmp"
                with open(_tmp, "w") as _fh:
                    _json.dump({"key": cache_key, "classes": _classes_out},
                               _fh, indent=1)
                os.replace(_tmp, str(width_cache_path))
            except Exception as _e:
                logger.warning("island width cache write failed: %s", _e)
        if cost_table_path is not None and cost_meta is not None:
            try:
                _dirty = False
                for pl in plans:
                    m = measured.get(int(pl.class_idx))
                    if m is None:
                        continue
                    _k = _sig + "|" + _wckey(pl)
                    _ent = cost_entries.setdefault(_k, {})
                    for _w, _c, _tag in m.get("probes", []):
                        # The table's contract is "the MAX over the WHOLE probe
                        # set at width w, measured autotune-OFF" -- that is what
                        # `_probe` reads back. effarc2 K13 added two tuple kinds
                        # that are NOT that, and letting them in mixes meters
                        # under one key: "+1k" is ONE kernel's demand (the
                        # single-kernel rung-2 probe) and "+prod" is the
                        # autotune-ON production measurement (~+33% by the
                        # `_xla_call_bytes` docstring's own accounting). A
                        # too-large entry at w=1 makes the next tile of the
                        # family take `c1 > budget_bytes` and route the whole
                        # class to the giant path (seed geometry, no island
                        # fit), i.e. a cached number would decide a SOLVER
                        # PATH. "+table" is
                        # excluded for the older reason: it is a read-back, not
                        # a measurement.
                        _t = str(_tag)
                        if _c is None or _t.endswith(
                                ("+table", "+1k", "+prod")):
                            continue
                        _old = _ent.get(str(int(_w)))
                        if _old is not None and int(_old[0]) != int(_c):
                            logger.warning(
                                "width cost table: entry mismatch for %s "
                                "w=%d (table %d vs fresh %d bytes) -- "
                                "keeping the fresh measurement", _k,
                                int(_w), int(_old[0]), int(_c))
                        if (_old is None or int(_old[0]) != int(_c)
                                or str(_old[1]) != str(_tag)):
                            _ent[str(int(_w))] = [int(_c), str(_tag)]
                            _dirty = True
                if _dirty:
                    os.makedirs(os.path.dirname(cost_table_path),
                                exist_ok=True)
                    _tmpc = cost_table_path + ".tmp"
                    with open(_tmpc, "w") as _fh:
                        _json.dump({"meta": cost_meta,
                                    "entries": cost_entries}, _fh, indent=1)
                    os.replace(_tmpc, cost_table_path)
            except Exception as _e:
                logger.warning("width cost table write failed: %s", _e)
        for pl in plans:
            m = measured[int(pl.class_idx)]
            if int(m["width"]) <= 0:
                # memory-infeasible even one island at a time -> giant
                # path: seed geometry, no island fit (measured replacement
                # for the byte model's per_island > budget downgrade)
                logger.warning(
                    "island stage: bucket bbox=%d nsrc=%d kind=%s tier=%d: "
                    "%d island(s) do not fit the joint LM one at a time "
                    "(%.1f GiB needed vs %.1f GiB free) -> giant path "
                    "(seed geometry)", pl.bbox_class, pl.nsrc_class,
                    pl.kind, pl.tier, pl.n_islands,
                    m["cost_bytes"] / 2 ** 30, m["budget_bytes"] / 2 ** 30)
                extra_giant_ids.extend(int(i) for i in pl.isl_ids)
                continue
            # LPT ordering unit: measured bytes/island x the kind's relative
            # compute cost (replaces the modelled per_island_bytes).
            _seed_class_tasks(
                pl, int(m["width"]),
                float(m["per_island_bytes"])
                * _COST_KIND_FACTOR.get(pl.kind, 1.0))
        pending.sort(key=lambda t: -t["cost"])
        n_tasks_a = len(pending)
        n_tbl = sum(1 for m in measured.values()
                    for p in m.get("probes", [])
                    if str(p[2]).endswith("+table"))
        logger.info(
            "grid_gpu[island]: chunk widths MEASURED for %d class(es) in "
            "%.1fs (%d probe(s), %d reused from cache, %d cost(s) from "
            "table; budget = free device memory, cost = XLA's own "
            "per-kernel report): %s", len(plans),
            time.perf_counter() - t_meas, n_probes, n_reused, n_tbl,
            ", ".join(
                f"({p.bbox_class},{p.nsrc_class},{p.kind},t{p.tier})"
                f"={measured[int(p.class_idx)]['width']}"
                f"/{p.n_islands}"
                for p in plans))

        running: dict[Any, tuple[int, dict]] = {}
        idle: set[int] = set(range(n_slots))
        slot_pref: dict[int, int] = {}

        def _pick(slot):
            if not pending:
                return None
            pref = slot_pref.get(slot)
            if pref is not None:
                for i, t in enumerate(pending):   # sorted desc by cost
                    if t["ci"] == pref:
                        return pending.pop(i)
            return pending.pop(0)

        def _submit(slot, task):
            fut = executors[slot].submit(
                _shard_run_chunk, task["ci"], task["rows"],
                task["width"], task["max_steps"], task["init_state"])
            slot_pref[slot] = task["ci"]
            running[fut] = (slot, task)

        def _fill_idle():
            for slot in sorted(idle):
                task = _pick(slot)
                if task is None:
                    break
                idle.discard(slot)
                _submit(slot, task)

        _fill_idle()
        while running:
            done, _ = futures_wait(set(running), return_when=FIRST_COMPLETED)
            # `done` is a SET of Future objects, so plain iteration is
            # heap-address order -- and the loop below accumulates
            # `tr["hard_rows"]`, whose ORDER decides which
            # islands share a phase-B chunk and in which vmap lane. Two
            # sources of run-to-run variation (completion timing and object
            # hashing) therefore reach a discrete decision. The author
            # already sorted the sibling set at `for slot in sorted(idle)`;
            # this is the same treatment. Iterating in submission order
            # costs nothing (the futures are all finished) and does not
            # touch scheduling -- `_fill_idle()` runs after the loop.
            for fut in _in_submit_order(done, running):
                slot, task = running.pop(fut)
                idle.add(slot)
                out = fut.result()
                busy[out["gpu"]] = busy.get(out["gpu"], 0.0) + out["wall_s"]
                ci = task["ci"]
                plan = plans[ci]
                tr = class_track[ci]
                rows = task["rows"]
                tr["res_p"][rows] = out["p"]
                tr["res_f"][rows] = out["fluxes"]
                tr["res_fe"][rows] = out["ferr"]
                tr["res_chi2"][rows] = out["chi2"]
                if out.get("n_good") is not None:
                    tr["res_ngood"][rows] = out["n_good"]
                tr["res_conv"][rows] = out["conv"]
                tr["res_stall"][rows] = out["stall"]
                tr["res_div"][rows] = out["diverged"]
                tr["res_pdchi2"][rows] = out.get(
                    "stall_dchi2",
                    np.full(len(rows), np.nan, np.float32))
                tr["res_railed"][rows] = out.get(
                    "railed", np.zeros(len(rows), bool))
                tr["res_nst"][rows] = out["nst"]
                if out.get("perr") is not None:
                    tr["res_perr"][rows] = out["perr"]
                if out.get("femarg") is not None:
                    tr["res_femarg"][rows] = out["femarg"]
                if out.get("smetrics") is not None:
                    tr["res_smetrics"][rows] = out["smetrics"]
                if out.get("swin") is not None:
                    tr["res_swin"][rows] = out["swin"]
                if out.get("reprof") is not None:
                    tr["res_reprof"][rows] = out["reprof"]
                tr["gpu_s"] += out["wall_s"]
                per_task_log.append(dict(
                    key=plan.key, phase=task["phase"], n=int(rows.size),
                    width=int(out.get("width_used", task["width"])),
                    gpu=out["gpu"],
                    wall_s=out["wall_s"], steps=int(out["steps_run"])))
                logger.info(
                    "island stage[shard]: bbox=%d nsrc=%d %s t%d "
                    "phase-%s chunk (%d isl, width %d) on GPU %s: %.1fs, "
                    "%d step(s)",
                    plan.bbox_class, plan.nsrc_class, plan.kind, plan.tier,
                    task["phase"], rows.size,
                    int(out.get("width_used", task["width"])), out["gpu"],
                    out["wall_s"], out["steps_run"])
                if task["phase"] == "A":
                    tr["main_gpu_s"] += out["wall_s"]
                    tr["outstanding"] -= 1
                    if budget_hard > 0:
                        uc = ~out["conv"]
                        if np.any(uc):
                            tr["hard_rows"].append(rows[uc])
                            # KEEP IN SYNC with island_solver._CARRY_KEYS
                            # (duplicated literal: this parent stays
                            # JAX-free, so it must not import the solver).
                            tr["hard_state"].append(
                                {k: out[k][uc] for k in
                                 ("p", "lam", "chi2", "consec", "nrej",
                                  "nst", "diverged", "n_same", "prej")})
                    if tr["outstanding"] == 0:
                        if tr["hard_rows"]:
                            # class main pass drained -> enqueue its
                            # compact hard chunks (LPT position by cost)
                            pending.extend(_make_hard_tasks(plan, tr))
                            pending.sort(key=lambda t: -t["cost"])
                        else:
                            _class_done_log(plan, tr)
                else:
                    tr["hard_gpu_s"] += out["wall_s"]
                    tr["hard_outstanding"] -= 1
                    if tr["hard_outstanding"] == 0 and tr["outstanding"] == 0:
                        _class_done_log(plan, tr)
            _fill_idle()

        makespan = time.perf_counter() - t_stage0

        # ---- #34 per-island residual export (opt-in) --------------------
        # One extra 0-step want_residual chunk per class at the converged
        # params (workers alive, field stacks on device). Mirrors the
        # single-process island_fit(return_island_residual=True) export;
        # keyed by GLOBAL island id with the (B,B) window field origin so
        # the host detector maps local peaks -> field pixel coords.
        island_residuals: dict | None = None
        residual_export_s = 0.0
        if export_residual:
            t_re = time.perf_counter()
            island_residuals = {}
            re_tasks: list[tuple] = []
            for plan in plans:
                ci = int(plan.class_idx)
                tr = class_track.get(ci)
                if tr is None:      # downgraded to the giant path
                    continue
                n_isl = plan.n_islands
                chunk = int(tr["chunk"])
                host = class_host_arrays(plan, **src_kwargs)
                for c0 in range(0, n_isl, chunk):
                    rows = np.arange(c0, min(c0 + chunk, n_isl),
                                     dtype=np.int64)
                    re_tasks.append(
                        (ci, rows, ladder_width(rows.size, chunk),
                         plan, host))

            re_running: dict[Any, tuple] = {}

            def _re_submit(slot, task):
                ci, rows, width, _plan, _host = task
                pp = class_track[ci]["res_p"][rows]
                zi = np.zeros(rows.size, np.int32)
                init = dict(
                    p=pp, lam=np.zeros(rows.size, np.float32),
                    chi2=np.zeros(rows.size, np.float32),
                    consec=zi, nrej=zi.copy(), nst=zi.copy())
                fut = executors[slot].submit(
                    _shard_run_chunk, ci, rows, int(width), 0, init,
                    True, False)   # want_residual=True, param_errors=False
                re_running[fut] = (slot, task)

            re_idle = list(range(n_slots))
            ti = 0
            while ti < len(re_tasks) and re_idle:
                _re_submit(re_idle.pop(), re_tasks[ti])
                ti += 1
            while re_running:
                done, _ = futures_wait(set(re_running),
                                       return_when=FIRST_COMPLETED)
                for fut in _in_submit_order(done, re_running):
                    slot, task = re_running.pop(fut)
                    ci, rows, _width, plan, host = task
                    out = fut.result()
                    for kk, r in enumerate(rows):
                        isl_id = int(plan.isl_ids[r])
                        island_residuals[isl_id] = dict(
                            resid=out["resid"][kk],
                            resid_w=out["resid_w"][kk],
                            x0b=float(host["x0b"][r]),
                            y0b=float(host["y0b"][r]),
                            B=int(plan.bbox_class))
                    if ti < len(re_tasks):
                        _re_submit(slot, re_tasks[ti])
                        ti += 1
                    else:
                        re_idle.append(slot)
            residual_export_s = time.perf_counter() - t_re
            logger.info(
                "grid_gpu[island]: #34 residual export -- %d island "
                "residual(s) over %d chunk(s) in %.1fs",
                len(island_residuals), len(re_tasks), residual_export_s)

        # clean shm detach in every worker before shutdown
        for ex in executors:
            try:
                ex.submit(_shard_close).result(timeout=60)
            except Exception:  # noqa: BLE001
                pass
    finally:
        for ex in executors:
            ex.shutdown(wait=True)

    # ---- assemble per-source rows (parent, NumPy only) ----
    compute_perr = bool(payload.get("compute_param_errors", False))
    compute_smetrics = bool(payload.get("compute_source_metrics", False))
    compute_fmarg = bool(payload.get("compute_flux_marg", False))
    compute_reprof = bool(payload.get("compute_re_profile", False))
    out_rows = []
    bucket_stats = []
    for plan in plans:
        tr = class_track.get(int(plan.class_idx))
        if tr is None:
            # measured memory-infeasible -> its islands were re-routed to
            # the giant path and are assembled by giant_seed_rows below
            continue
        host = class_host_arrays(plan, **src_kwargs)
        out_rows.append(unpack_class_results(
            plan, host, island_set,
            tr["res_p"], tr["res_f"], tr["res_fe"], tr["res_chi2"],
            tr["res_conv"], tr["res_stall"], tr["res_nst"],
            res_perr=tr["res_perr"] if compute_perr else None,
            res_reprof=tr["res_reprof"] if compute_reprof else None,
            reprof_deltas=REPROF_DELTAS_LN,
            res_smetrics=tr["res_smetrics"] if compute_smetrics else None,
            res_swin=tr["res_swin"] if compute_smetrics else None,
            res_femarg=tr["res_femarg"] if compute_fmarg else None,
            res_div=tr["res_div"],
            res_pdchi2=tr["res_pdchi2"],
            res_railed=tr["res_railed"],
            res_ngood=tr["res_ngood"], n_bands=n_bands))
        n_isl = plan.n_islands
        dt = tr["gpu_s"]
        bucket_stats.append(dict(
            bbox_class=plan.bbox_class, nsrc_class=plan.nsrc_class,
            tier=plan.tier, kind=plan.kind,
            n_islands=int(n_isl), n_chunks=tr["n_chunks"],
            chunk=tr["chunk"], wall_s=dt,
            islands_per_s=n_isl / dt if dt > 0 else float("inf"),
            mean_steps=float(tr["res_nst"].mean()),
            max_steps_used=int(tr["res_nst"].max()),
            n_unconverged=int((~tr["res_conv"]).sum()),
            n_stalled=int(tr["res_stall"].sum()),
            n_requeued=tr["n_hard"],
            hard_chunks=tr["hard_chunks"], hard_chunk=tr["hard_chunk"],
            hard_wall_s=tr["hard_gpu_s"], main_wall_s=tr["main_gpu_s"],
        ))
    # Giant islands (true giants, S5-capped, memory-infeasible classes):
    # seed geometry, no island fit; pass 2 measures their fluxes (R16).
    # giant_seed_rows sorts by island id, so the row order is
    # run-independent (development ledger SS24).
    out_rows.extend(giant_seed_rows(
        np.concatenate([
            np.asarray(giant_ids, dtype=np.int64).ravel(),
            np.asarray(extra_giant_ids, dtype=np.int64).ravel()]),
        island_set, pos_x=src_kwargs["pos_x"], pos_y=src_kwargs["pos_y"],
        logre=src_kwargs["logre"], ee1=src_kwargs["ee1"],
        ee2=src_kwargs["ee2"], sersic_n=src_kwargs["sersic_n"],
        n_bands=n_bands))

    cat = {k: np.concatenate([r[k] for r in out_rows], axis=0)
           for k in out_rows[0]}
    ires = IslandFitResult(
        bandnames=list(payload["bandnames"]),
        source_idx=cat["src"].astype(np.int64),
        pos_x=cat["pos_x"], pos_y=cat["pos_y"],
        logre=cat["logre"], ee1=cat["ee1"], ee2=cat["ee2"],
        sersic_n=cat["sersic_n"],
        fluxes=cat["fluxes"], flux_err=cat["flux_err"],
        chi2=cat["chi2"], chi2_nu=cat["chi2_nu"], n_steps=cat["n_steps"],
        converged=cat["converged"], stalled=cat["stalled"],
        diverged=cat["diverged"],
        stall_dchi2=cat.get("stall_dchi2"),
        railed=cat["railed"],
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
    total_busy = sum(busy.values())
    n_requeued = int(sum(s["n_requeued"] for s in bucket_stats))
    hard_gpu_s = float(sum(s["hard_wall_s"] for s in bucket_stats))
    shard_stats = dict(
        makespan_s=makespan,
        per_gpu_busy_s={k: round(v, 1) for k, v in sorted(busy.items())},
        total_busy_s=total_busy,
        ideal_makespan_s=total_busy / max(1, n_slots),
        n_tasks=n_tasks_a + sum(s["hard_chunks"] for s in bucket_stats),
        n_requeued_islands=n_requeued,
        hard_gpu_s=hard_gpu_s,
        residual_export_s=residual_export_s,
        budget_main=budget_main, budget_hard=budget_hard,
        per_task=per_task_log,
    )
    return ires, shard_stats, island_residuals


# ---------------------------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------------------------


def run_island_prestage(
    *,
    tasks: list,                       # pass-1 PerGridTask list (S1-only)
    grids: list,                       # GridDef list (for core regions)
    devices: list[int],
    gpu_cfg: dict,
    island_cfg: dict,
    bandnames_with_psf: list[str],
    psfs_per_band_objs: list,
    per_band_full: dict,               # bn -> (data, invvar, pixg); may be empty (shm mode)
    shared_full_specs: dict,           # bn -> ((name, shape, dtype) x3); may be empty
    image_hw: tuple[int, int],
    keep_idx: np.ndarray,              # global indices of fit sources
    n_total: int,
    xpix: np.ndarray, ypix: np.ndarray,
    source_type: np.ndarray,
    logre_init: np.ndarray, ee1_init: np.ndarray, ee2_init: np.ndarray,
    sersic_n_init: np.ndarray,
    render_backend: str,
    fit_sersic_n: bool,
    n_fit_min_snr: float,
    pos_max_shift_pix: float,
    # Catalog-driven Gaussian PRIOR sigma columns (full-catalog length,
    # NaN where absent/invalid -> no prior). See class_host_arrays.
    pos_err_prior: np.ndarray | None = None,        # (n_total,) POS_ERR (px)
    re_prior_sig: np.ndarray | None = None,         # (n_total,) Re_PRIOR_SIG (px)
    ell_prior_sig: np.ndarray | None = None,        # (n_total,) ELL_PRIOR_SIG
    theta_prior_sig: np.ndarray | None = None,      # (n_total,) THETA_PRIOR_SIG (deg)
    sersic_n_prior_sig: np.ndarray | None = None,   # (n_total,) SERSIC_N_PRIOR_SIG
    bg_x_origin: float, bg_x_scale: float,
    bg_y_origin: float, bg_y_scale: float,
    bg_poly_order: int,
    jit_cache_dir,
    worker_log_file,
    dispatch_fn,                       # pipeline._run_grids_on_slots
    dispatch_packed_fn=None,           # effarc2 K7: packed pass-1 dispatch
    bright_mask_det: np.ndarray | None = None,  # (n,3) unmatched (x,y,flux) -> DET
    work_dir: str | None = None,       # run work dir (subtract_detected.csv)
    bright_mask_sub: np.ndarray | None = None,  # (n,3) Exp C subtract-routed (x,y,flux)
    bright_mask_sub_nsigma: float | None = None,  # their detection significance
    phantom_mask: np.ndarray | None = None,  # (n_total,) Exp C v2 phantom rows
) -> tuple[list, dict]:
    """Run passes 1 + island stage; return (pass-2 tasks, island meta).

    ``island meta`` is a dict of full-catalog-length per-source arrays
    (island_id / island_nsrc / island_giant / island_stalled /
    island_diverged (#35) / converged / n_steps / free_n / fitted_mask,
    plus the #35 ``flux_win`` (n_total, n_bands) window fluxes with their
    band order ``flux_win_bands``, plus the #28a error
    arrays pos_x_err / pos_y_err / re_err / ell_err / theta_err /
    sersic_n_err -- all-NaN when
    ``compute.gpu.error_model.param_errors_enabled`` is false) plus the
    updated positions/shapes, consumed by the pipeline's post-merge
    writeback.
    """
    import multiprocessing as mp

    from multiprocessing import shared_memory

    from .bg_polynomial import make_bg_basis
    from .psf_polynomial import evaluate_psf_at_np

    kappa = float(island_cfg.get("kappa", 0.5))
    max_steps = int(island_cfg.get("max_steps", 30))
    # #28a knob: post-convergence marginal nonlinear-parameter errors
    # (one extra read-only Hessian evaluation per chunk; default on).
    param_errors = bool((gpu_cfg.get("error_model", {}) or {})
                        .get("param_errors_enabled", True))
    # #17 knob: per-source model-quality metrics (rchisq/fracflux/fracin/
    # fracmasked) at convergence -- one extra read-only band scan per chunk;
    # default on. An off run is byte-identical (kernel not called, columns
    # absent).
    source_metrics = bool((gpu_cfg.get("error_model", {}) or {})
                          .get("source_metrics_enabled", True))
    # D2 knob: SHAPE-marginalized flux error (root-fix D). One extra
    # read-only Hessian-inverse + VarPro-Jacobian evaluation per island
    # chunk at convergence; default on. An off run is byte-identical (the
    # kernel is not called and the FLUXERR_MARG* columns are absent).
    marginal_flux_err = bool((gpu_cfg.get("error_model", {}) or {})
                             .get("marginal_flux_err_enabled", True))
    # R1: whether the catalog supplied ANY prior width (gates the
    # prior_frac disclosure OUTPUT columns; a no-prior run is
    # byte-identical because the columns are absent entirely).
    priors_active = bool(any(
        a is not None and np.any(np.isfinite(a))
        for a in (pos_err_prior, re_prior_sig, ell_prior_sig,
                  theta_prior_sig, sersic_n_prior_sig)))
    _msm = island_cfg.get("max_steps_main", DEFAULT_MAX_STEPS_MAIN)
    max_steps_main = int(_msm) if _msm is not None else 0
    joint_nsrc_cap = int(island_cfg.get("joint_nsrc_cap", 32))
    # PERF #14: mixed-class star/galaxy render split. Default TRUE for the
    # 2.2-2.6x speedup on mixed island buckets (dominant island-stage cost);
    # adds the split's fp-noise (same class as the existing chunk-width
    # nondeterminism). Set false for the byte-reproducible pre-#14 path.
    star_render_split = bool(island_cfg.get("star_render_split", True))
    # #35: stall counter semantics. "proposals" (default) = legacy (every
    # rejected LM step counts toward the 4-reject stall); "changed" (opt-in
    # semantic fix) = a reject counts only when the clip/bound-limited
    # proposal differs from the previously rejected one (identical-proposal
    # rejects keep escalating lambda without counting). Static compiled-
    # kernel option; see docs/configuration.md (island.stall_count_mode)
    # for the measured displaced-valley caution.
    stall_count_mode = str(island_cfg.get(
        "stall_count_mode", "proposals")).strip().lower()
    if stall_count_mode not in ("proposals", "changed"):
        raise ValueError(
            "compute.gpu.solver.island.stall_count_mode must be "
            f"'proposals' or 'changed', got {stall_count_mode!r}")
    # Trust-region LM step (opt-in). "clip" (default) = the byte-identical
    # legacy damped-Newton + fixed-clip step; "trust_region" = scaled dogleg
    # trust region (robustness / determinism / stall-conversion; NOT accuracy
    # on displaced valleys). See docs/configuration.md (island.step_mode).
    step_mode = str(island_cfg.get(
        "step_mode", "clip")).strip().lower()
    if step_mode not in ("clip", "trust_region"):
        raise ValueError(
            "compute.gpu.solver.island.step_mode must be 'clip' or "
            f"'trust_region', got {step_mode!r}")
    # dlnp campaign (2026-07-30): in-loop convergence criteria. "legacy"
    # (default) = the byte-identical pre-dlnp step programs; "dlnp" =
    # f64 chi2 carry + absolute statistical ftol (the 1-sigma quantum)
    # + quantum-gated step tolerances + distinct-proposal stall counting
    # + frozen-at-cap terminal exit. Static compiled-kernel option; see
    # docs/configuration.md (island.convergence).
    convergence = str(island_cfg.get(
        "convergence", "legacy")).strip().lower()
    if convergence not in ("legacy", "dlnp"):
        raise ValueError(
            "compute.gpu.solver.island.convergence must be 'legacy' or "
            f"'dlnp', got {convergence!r}")
    if convergence == "dlnp" and step_mode != "clip":
        raise ValueError(
            "compute.gpu.solver.island.convergence: 'dlnp' requires "
            "step_mode: 'clip' (trust-region support pending)")
    if convergence == "dlnp" and stall_count_mode == "changed":
        raise ValueError(
            "compute.gpu.solver.island.convergence: 'dlnp' subsumes "
            "stall_count_mode: 'changed' (distinct-proposal counting is "
            "the dlnp behavior); leave stall_count_mode at 'proposals'")
    plateau_dlnp_tol = float(island_cfg.get("plateau_dlnp_tol", 0.5))
    # Scene subtraction (#35 root fix, opt-in): render every non-member
    # cataloged source at its S1 flux and subtract it from each island cutout
    # (as the exact-equivalent field-wide subtract + in-kernel member add-back).
    # Off => byte-identical (path skipped). See docs/configuration.md.
    scene_subtract = bool(island_cfg.get("scene_subtract", False))
    # Exp B footprint protection (always on; the opt-out
    # island.subtract_detected_protect and the SEP subtract_detected pass
    # were removed in card 018): never subtract a subtract-only candidate
    # (Exp C bright-mask-routed detections, phantom columns) that a
    # cataloged extended source's own S1 model can explain.
    # Item-4 Layer-2 blend-resolve (opt-in, staged): find the uncataloged
    # point source a blend-suspect host absorbed, by scanning the S1-STATE
    # residual (the state where positions are frozen at seeds, so the star
    # is not yet re-explained by the displaced host -- oracle: 46-150 sigma
    # matched-filter there vs invisible post-fit). mode "detect" = pure
    # disclosure (blend_candidates.csv), fit untouched. Requires
    # scene_subtract (the detector consumes the scene-cleaned dmb stack).
    # enabled=false skips the whole path -> byte-identical.
    br_cfg = island_cfg.get("blend_resolve", {}) or {}
    br_on = bool(br_cfg.get("enabled", False))
    br_mode = str(br_cfg.get("mode", "detect")).strip().lower()
    br_nsigma = float(br_cfg.get("nsigma", 5.0))
    br_minarea = int(br_cfg.get("minarea", 5))
    if br_on and br_mode not in ("detect", "resolve"):
        raise ValueError(
            "compute.gpu.solver.island.blend_resolve.mode must be "
            f"'detect' or 'resolve' (got {br_mode!r})")
    if br_on and not scene_subtract:
        logger.warning(
            "blend_resolve requires island.scene_subtract; disabling "
            "the blend-resolve detector for this run")
        br_on = False
    if max_steps_main <= 0 or max_steps_main >= max_steps:
        budget_main, budget_hard = max_steps, 0
    else:
        budget_main, budget_hard = max_steps_main, max_steps - max_steps_main
    n_bands = len(bandnames_with_psf)
    H, W = int(image_hw[0]), int(image_hw[1])
    native = int(psfs_per_band_objs[0].H_psf)
    mem_fraction = float(gpu_cfg.get("mem_fraction", 0.5))
    ctx = mp.get_context("spawn")

    # ------------------------------------------------------------------
    # Pass 1: per-grid S1 seed solve (positions/shapes at catalog seeds)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    logger.info(
        "grid_gpu[island]: pass 1 -- per-grid S1 seed solve, %d grid(s) "
        "on %d GPU worker(s)", len(tasks), len(devices))
    if dispatch_packed_fn is not None:
        # effarc2 K7: same measured packing / MPS / OOM-retry the final grid
        # pass uses. Scheduling-only: per-grid math is independent and merged
        # by source_indices.
        results = dispatch_packed_fn(tasks, list(devices), ctx=ctx)
    else:
        results, _ = dispatch_fn(
            tasks, list(devices), ctx, "platform", mem_fraction,
            jit_cache_dir, worker_log_file, False,
        )
    failed = [r for r in results if getattr(r, "error", None)]
    if failed:
        raise RuntimeError(
            f"grid_gpu[island]: {len(failed)} grid(s) failed the pass-1 "
            f"S1 seed solve; island mode cannot proceed without seed "
            f"fluxes + background planes: "
            f"{[(r.grid_id, str(r.error)[:80]) for r in failed[:5]]}"
        )
    logger.info("grid_gpu[island]: pass 1 done in %.1fs",
                time.perf_counter() - t0)

    # Seed fluxes/vars per global source row (each fit source is owned
    # by exactly one grid).
    s1_flux = np.full((n_total, n_bands), np.nan, dtype=np.float32)
    s1_var = np.full((n_total, n_bands), np.nan, dtype=np.float32)
    for r in results:
        s1_flux[r.source_indices] = r.fluxes
        s1_var[r.source_indices] = r.flux_var

    # ------------------------------------------------------------------
    # Field-level band views (grid-stage shm or in-RAM dict).
    # ------------------------------------------------------------------
    _keepalive: list[Any] = []

    def _band_view(bi: int) -> tuple[np.ndarray, np.ndarray]:
        bn = bandnames_with_psf[bi]
        if bn in per_band_full:
            data, invvar, _pixg = per_band_full[bn]
            return data, invvar
        specs = shared_full_specs[bn]
        out = []
        for spec in specs[:2]:              # (data, invvar)
            name, shape, dtype = spec
            shm = shared_memory.SharedMemory(name=name)
            _keepalive.append(shm)
            out.append(np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf))
        return out[0], out[1]

    t0 = time.perf_counter()
    band_views = [_band_view(bi) for bi in range(n_bands)]
    sigma_bands: list[float] = []
    for bi in range(n_bands):
        sub = band_views[bi][1][::16, ::16]
        good = sub > 0
        if np.any(good):
            sigma_bands.append(
                float(np.median(1.0 / np.sqrt(sub[good]))))
    sigma_sky = float(np.median(sigma_bands))

    # ------------------------------------------------------------------
    # Island decomposition (host, numpy) from the S1 fluxes + class
    # planning (chunk widths, giant capping) -- all JAX-free.
    # ------------------------------------------------------------------
    kx = xpix[keep_idx].astype(np.float64)
    ky = ypix[keep_idx].astype(np.float64)
    kst = source_type[keep_idx].astype(np.int32)
    klr = logre_init[keep_idx].astype(np.float32)
    ke1 = ee1_init[keep_idx].astype(np.float32)
    ke2 = ee2_init[keep_idx].astype(np.float32)
    ksn = sersic_n_init[keep_idx].astype(np.float32)
    kflux = s1_flux[keep_idx]
    kvar = s1_var[keep_idx]
    # Exp C v2 phantom rows within the fit set (original, pre-DET-append
    # length; DET rows appended in later rounds are never phantoms).
    kph = (np.asarray(phantom_mask, dtype=bool)[keep_idx]
           if phantom_mask is not None else None)
    demoted_glob: list[int] = []   # protection-demoted phantom global rows
    ref_band = len(bandnames_with_psf) // 2
    stamp = evaluate_psf_at_np(
        psfs_per_band_objs[ref_band], W / 2.0, H / 2.0)
    psf_r, psf_p = psf_radial_profile(np.asarray(stamp, dtype=float))
    r_psf = native / 2.0

    # ---- DET members (bright_mask DET-routing, "model the wings") ----
    # Unmatched bright-mask detections join the FIT SET as marked DET
    # members (compact Sersic seed, n FROZEN, Re ~ 1 px, round) so their
    # light is attributed to their own model instead of a neighbour's
    # wings; DET rows are dropped at fold-back (not emitted). The seed and
    # the seed-variance S/N (5) are the values the removed
    # island.detect_unattributed block supplied by default in dev8
    # (source_type "sersic", nsigma 5.0; card 018).
    _DET_SEED_NSIGMA = 5.0

    def _append_det(kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar, dpx, dpy, dfl):
        """Append DET peaks as marked FIT-SET members (compact, n-frozen);
        returns the augmented arrays + n_det + fit_to_global (-1 for DET)."""
        n_new = int(dpx.size)
        if n_new == 0:
            return (kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar, 0, keep_idx)
        from .source_model import SOURCE_SERSIC as _SRC_SERSIC_DET
        shp = dict(source_type=int(_SRC_SERSIC_DET), logre=float(np.log(1.0)),
                   ee1=0.0, ee2=0.0, sersic_n=1.0)
        kx = np.concatenate([kx, dpx.astype(np.float64)])
        ky = np.concatenate([ky, dpy.astype(np.float64)])
        kst = np.concatenate([kst, np.full(n_new, shp["source_type"], np.int32)])
        klr = np.concatenate([klr, np.full(n_new, shp["logre"], np.float32)])
        ke1 = np.concatenate([ke1, np.full(n_new, shp["ee1"], np.float32)])
        ke2 = np.concatenate([ke2, np.full(n_new, shp["ee2"], np.float32)])
        ksn = np.concatenate([ksn, np.full(n_new, shp["sersic_n"], np.float32)])
        f_seed = np.maximum(dfl.astype(np.float32), 1e-6)
        v_seed = (f_seed / max(_DET_SEED_NSIGMA, 3.0)) ** 2
        kflux = np.concatenate(
            [kflux, np.tile(f_seed[:, None], (1, n_bands))], axis=0)
        kvar = np.concatenate(
            [kvar, np.tile(v_seed[:, None], (1, n_bands))], axis=0)
        fit_to_global = np.concatenate(
            [keep_idx, np.full(n_new, -1, dtype=keep_idx.dtype)])
        return (kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar, n_new,
                fit_to_global)

    # ------------------------------------------------------------------
    # Reusable round: island decomposition (host, numpy) + class planning
    # + PADDED field stacks + chunk-sharded island fit. Called once for
    # the OFF / RAW-one-pass paths; up to twice for the residual path
    # (round 0 exports residuals for detection, round 1 refits with the
    # DET members). `_keepalive` (the grid-stage band-view shm handles) is
    # SHARED across rounds -> closed by the caller AFTER the last round,
    # never here.
    # ------------------------------------------------------------------
    def _decompose_and_fit(kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar,
                           *, n_det, want_residual, round_idx):
        t0r = time.perf_counter()
        with np.errstate(invalid="ignore"):
            flux_med = np.nanmedian(kflux, axis=1)
        radii = flux_aware_radii(
            kst, klr, ksn, flux_med, sigma_sky, kappa, psf_r, psf_p, r_psf)
        # Exp C v2: phantom rows are FORCED SINGLETON islands -- they may
        # never joint-fit with (or merge into) a real source's island;
        # their light reaches other islands only via the scene-subtract
        # field pass at their jointly-solved S1 flux. DET rows appended in
        # later rounds are never phantoms (pad with False).
        kph_pad = None
        if kph is not None and kph.any():
            kph_pad = np.zeros(kx.size, dtype=bool)
            kph_pad[:kph.size] = kph
        island_set = build_islands(kx, ky, kst, klr, ksn, r_psf,
                                   radius=radii,
                                   forced_singleton=kph_pad)
        buckets = bucket_islands(island_set, ISLAND_BBOX_CLASSES,
                                 ISLAND_NSRC_CLASSES)
        occ = {k: b.n_islands for k, b in sorted(buckets.buckets.items())}
        logger.info(
            "grid_gpu[island]: round %d: %d islands from %d fit sources "
            "(%d DET) in %.1fs (kappa=%.2g, ref band %s); nsrc "
            "p50/p90/max = %d/%d/%d; giants=%d; buckets=%s",
            round_idx, island_set.n_islands, kx.size, n_det,
            time.perf_counter() - t0r, kappa, bandnames_with_psf[ref_band],
            int(np.percentile(island_set.n_src, 50)),
            int(np.percentile(island_set.n_src, 90)),
            int(island_set.n_src.max()), buckets.n_giant, occ)

        plans, capped_giant_ids = plan_island_classes(
            island_set, buckets, kst, klr, ksn, native,
            joint_nsrc_cap=joint_nsrc_cap,
            log_fn=lambda m: logger.info("island stage: %s", m))
        giant_ids = np.concatenate([
            np.asarray(buckets.giant_ids, dtype=np.int64).ravel(),
            np.asarray(capped_giant_ids, dtype=np.int64),
        ]).astype(np.int64)

        # Production free-n gate on the SAME sanitized S1 arrays the solver
        # sees (nan -> 0 flux / huge var), so parent and workers agree.
        ks1_flux = np.nan_to_num(kflux, nan=0.0)
        ks1_var = np.nan_to_num(kvar, nan=1e30)
        free_n_all = free_n_gate(kst, ks1_flux, ks1_var,
                                 bool(fit_sersic_n), float(n_fit_min_snr))
        if n_det:
            # #34: DET rows are seeds, not beliefs -- FREEZE their Sersic n
            # so a detected blob can never rail its own n (seeded compact).
            free_n_all[-n_det:] = False

        # PADDED field stacks: data - frozen per-grid S1 background, +
        # invvar (border of PAD zeros so island windows never clip). Round-
        # unique shm names so a residual round's stacks never collide.
        t0s = time.perf_counter()
        PAD = int(max((p.bbox_class for p in plans), default=32)) + 8
        Hp, Wp = H + 2 * PAD, W + 2 * PAD
        pid = os.getpid()
        shm_handles: list[Any] = []

        def _new_shm(tag: str) -> tuple[Any, np.ndarray]:
            seg = shared_memory.SharedMemory(
                create=True, size=n_bands * Hp * Wp * 4,
                name=f"tract7dt_island_{tag}_{pid}_r{round_idx}",
            )
            shm_handles.append(seg)
            arr = np.ndarray((n_bands, Hp, Wp), dtype=np.float32,
                             buffer=seg.buf)
            arr[:] = 0.0
            return seg, arr

        try:
            dmb_seg, dmb = _new_shm("data")
            iv_seg, iv = _new_shm("invvar")

            res_by_gid = {int(r.grid_id): r for r in results}
            grid_by_gid = {int(g.grid_id): g for g in grids}
            for bi in range(n_bands):
                data_b, invvar_b = band_views[bi]
                dmb[bi, PAD:PAD + H, PAD:PAD + W] = data_b
                iv[bi, PAD:PAD + H, PAD:PAD + W] = invvar_b
            for gid, r in res_by_gid.items():
                g = grid_by_gid[gid]
                h = g.y1_core - g.y0_core
                w = g.x1_core - g.x0_core
                basis = make_bg_basis(
                    h, w, g.x0_core, g.y0_core,
                    bg_x_origin, bg_x_scale, bg_y_origin, bg_y_scale,
                    order=int(bg_poly_order),
                )
                for bi in range(n_bands):
                    dmb[bi, PAD + g.y0_core:PAD + g.y1_core,
                        PAD + g.x0_core:PAD + g.x1_core] -= (
                        basis @ np.asarray(r.bg_coefs[bi], dtype=np.float32)
                    ).reshape(h, w)
            # Scene subtraction (#35, opt-in): subtract EVERY fit-set source's
            # S1 model from the padded field stack (field-wide, grid-free). The
            # per-island fit ADDS BACK its own members in-kernel (run_class_chunk
            # after the gather), so the net effect per island cutout is removal
            # of NON-member wing light. render_source_stamps is renderer-
            # consistent with the kernel models_band (dx=frac(pos), raw-PSF for
            # stars); field-frame top-left = round(pos)+PAD-T//2 matches the
            # window x0b=ox-PAD convention -> members cancel exactly (stars) /
            # to ulp (galaxies, PERF#14 fp class). band_views is RAW (wing-peel
            # lives only in the per-grid S1 copy), so no giant double-subtract.
            if scene_subtract:
                import jax.numpy as _jnp
                from .forced_phot_sparse import render_source_stamps
                from .source_model import SOURCE_STAR as _SRC_STAR
                t0ss = time.perf_counter()
                # Timeline marker (perf breakdown): opens the scene-subtract
                # segment; the closing line below self-reports the total.
                logger.info(
                    "grid_gpu[island]: round %d: scene-subtract starting -- "
                    "%d fit sources x %d bands",
                    round_idx, int(kx.size), n_bands)
                Tt = int(native)
                # SUBTRACT-ONLY uncataloged candidates: bright-mask-routed
                # unmatched detections (Exp C) are cross-matched to the catalog
                # and the UNMATCHED kept as subtract-only STARS: their wing
                # contamination is removed field-wide but they NEVER enter
                # islanding / the joint fit (fit stays catalog-only).
                _sub_x = _sub_y = None
                _sub_thr = None        # per-candidate detection significance
                _det_rows = None       # persisted detection table (Exp B/C)
                _sub_rows = None       # row indices of _sub_x in _det_rows
                _det_num = None        # extended-model white numerator/det
                _has_bm = (bright_mask_sub is not None
                           and len(bright_mask_sub) > 0)
                _has_ph = kph_pad is not None
                if _has_bm or _has_ph:
                    from scipy.spatial import cKDTree as _ckd
                    _det_thresh = 8.0
                    _ivc = iv[:, PAD:PAD + H, PAD:PAD + W]
                    _ivs = _ivc.sum(0)
                    _den = np.where(_ivs > 0, _ivs, 1.0)
                    _nz = np.where(_ivs > 0, 1.0 / np.sqrt(_den), 1e30).astype(np.float32)
                    # Cross-match reference: FULL catalog (xpix/ypix) UNION
                    # the current fit set (kx/ky, which adds any DET sources
                    # appended in later rounds) -- NOT the fit set alone: a
                    # cataloged source dropped from the fit by the bright
                    # mask is still CATALOGED, and matching only the fit set
                    # made such a star look "uncataloged" and get
                    # mis-subtracted from an off-center re-detection
                    # (measured: oracle OSTK-0067 at 0.37x its true flux).
                    _catf = np.column_stack([xpix, ypix])
                    _catf = _catf[np.isfinite(_catf).all(1)]
                    _catk = np.column_stack([kx, ky])
                    _catk = _catk[np.isfinite(_catk).all(1)]
                    _ctree = _ckd(np.concatenate([_catf, _catk], axis=0))
                    _tx = []; _ty = []; _tdist = []; _tmatch = []
                    _tsrc = []; _tthr = []
                    if _has_bm:
                        # Exp C: bright-mask-routed unmatched detections
                        # ("subtract, don't mask"). Re-run the (stricter,
                        # union) cross-match for consistency.
                        _bx = np.asarray(bright_mask_sub[:, 0], np.float64)
                        _by = np.asarray(bright_mask_sub[:, 1], np.float64)
                        _bthr = float(bright_mask_sub_nsigma
                                      if bright_mask_sub_nsigma else
                                      _det_thresh)
                        _bdist, _ = _ctree.query(
                            np.column_stack([_bx, _by]), k=1)
                        _bdup = np.zeros(len(_bx), bool)
                        _kb = ~_bdup
                        _tx.extend(_bx[_kb]); _ty.extend(_by[_kb])
                        _tdist.extend(_bdist[_kb].astype(np.float64))
                        _tmatch.extend(list(_bdist[_kb] <= 3.0))
                        _tsrc.extend(["bmask"] * int(_kb.sum()))
                        _tthr.extend([_bthr] * int(_kb.sum()))
                    if _tx:
                        # Full candidate table -> {work_dir}/subtract_
                        # detected.csv (attribution: matched / protected /
                        # subtracted + per-band fluxes + src pass).
                        _det_rows = {
                            "x": np.asarray(_tx, np.float64),
                            "y": np.asarray(_ty, np.float64),
                            "dist_to_catalog_px": np.asarray(
                                _tdist, np.float64),
                            "matched": np.asarray(_tmatch, bool),
                            "protected": np.zeros(len(_tx), bool),
                            "src": np.asarray(_tsrc, object),
                        }
                        _um_all = ~_det_rows["matched"]
                        _sub_rows = np.nonzero(_um_all)[0]
                        _sub_x = _det_rows["x"][_sub_rows]
                        _sub_y = _det_rows["y"][_sub_rows]
                        _sub_thr = np.asarray(
                            _tthr, np.float64)[_sub_rows]
                _isext = kst != _SRC_STAR
                _lr = np.where(_isext, klr, 0.0).astype(np.float32)
                _e1 = np.where(_isext, ke1, 1.0).astype(np.float32)
                _e2 = np.where(_isext, ke2, 0.0).astype(np.float32)
                _ixt = np.round(kx).astype(np.int64) - Tt // 2 + PAD
                _iyt = np.round(ky).astype(np.int64) - Tt // 2 + PAD
                # Exp B footprint protection: map each unmatched detection
                # to the cataloged EXTENDED sources whose render stamp
                # covers it, then accumulate (inside the render loop below)
                # the extended-only white-stack model value at the
                # detection pixel with the SAME iv weighting as the
                # detection coadd _wh. A detection whose extended-model
                # white value ALONE clears the SEP threshold
                # (_det_thresh x local noise) is attributable to a
                # galaxy's own light -> excluded from the subtract-only
                # pass (self-subtraction guard). Derived rule: no new
                # constants, reuses the detection threshold + noise map.
                _det_by_src = None
                if (_sub_x is not None
                        and len(_sub_x) and bool(_isext.any())):
                    _det_pxp = np.round(_sub_x).astype(np.int64) + PAD
                    _det_pyp = np.round(_sub_y).astype(np.int64) + PAD
                    _det_num = np.zeros(len(_sub_x), np.float64)
                    _dtree = _ckd(np.column_stack([_sub_x, _sub_y]))
                    _det_by_src = {}
                    for _gk in np.nonzero(_isext)[0]:
                        if not (np.isfinite(kx[_gk]) and np.isfinite(ky[_gk])):
                            continue
                        _cand = _dtree.query_ball_point(
                            [float(kx[_gk]), float(ky[_gk])],
                            r=float(Tt) * 0.7072)
                        if _cand:
                            _det_by_src[int(_gk)] = _cand
                # Exp C v2: footprint-protection DEMOTION of phantom
                # columns (round 0, post-S1). Same physical test as
                # candidate protection: if the cataloged EXTENDED
                # sources' own S1 seed-shape models ALONE clear the
                # routing pass's detection threshold at the phantom's
                # position, the "detection" is a galaxy's own light.
                # Demoted phantom: S1 flux zeroed (no field subtraction,
                # no addback -- the galaxy's own model keeps that light)
                # and excluded from the pass-2 final solve via
                # demoted_glob (so it cannot absorb galaxy flux in the
                # reported fluxes). Also logs the G3 sign-flip monitor
                # (fraction of significantly negative phantom S1 fluxes).
                if (_has_ph and round_idx == 0
                        and bool(_isext.any())):
                    _phi = np.nonzero(kph_pad)[0]
                    _phxp = np.round(kx[_phi]).astype(np.int64) + PAD
                    _phyp = np.round(ky[_phi]).astype(np.int64) + PAD
                    _phcx = np.clip(
                        np.round(kx[_phi]).astype(np.int64), 0, W - 1)
                    _phcy = np.clip(
                        np.round(ky[_phi]).astype(np.int64), 0, H - 1)
                    _ph_num = np.zeros(len(_phi), np.float64)
                    _ptree = _ckd(np.column_stack([kx[_phi], ky[_phi]]))
                    _pby = {}
                    for _gk in np.nonzero(_isext)[0]:
                        if not (np.isfinite(kx[_gk])
                                and np.isfinite(ky[_gk])):
                            continue
                        _cand = _ptree.query_ball_point(
                            [float(kx[_gk]), float(ky[_gk])],
                            r=float(Tt) * 0.7072)
                        if _cand:
                            _pby[int(_gk)] = _cand
                    if _pby:
                        _eidx = np.fromiter(_pby.keys(), np.int64)
                        for bi in range(n_bands):
                            _pj = psfs_per_band_objs[bi].to_jax()
                            for s0 in range(0, len(_eidx), 512):
                                _sl = _eidx[s0:s0 + 512]
                                _st = np.asarray(render_source_stamps(
                                    _pj,
                                    _jnp.asarray(kx[_sl].astype(np.float32)),
                                    _jnp.asarray(ky[_sl].astype(np.float32)),
                                    0, 0,
                                    _jnp.asarray(kst[_sl].astype(np.int32)),
                                    _jnp.asarray(_lr[_sl]),
                                    _jnp.asarray(_e1[_sl]),
                                    _jnp.asarray(_e2[_sl]),
                                    sersic_n=_jnp.asarray(
                                        ksn[_sl].astype(np.float32)),
                                    render_backend=render_backend))
                                for _j, _gk in enumerate(_sl):
                                    _f = float(ks1_flux[_gk, bi])
                                    if _f == 0.0:
                                        continue
                                    _ixk = int(_ixt[_gk])
                                    _iyk = int(_iyt[_gk])
                                    for _di in _pby[int(_gk)]:
                                        _sy = int(_phyp[_di]) - _iyk
                                        _sx = int(_phxp[_di]) - _ixk
                                        if 0 <= _sy < Tt and 0 <= _sx < Tt:
                                            _ph_num[_di] += (
                                                float(iv[bi, _phyp[_di],
                                                         _phxp[_di]])
                                                * _f
                                                * float(_st[_j, _sy, _sx]))
                    _phden = _ivs[_phcy, _phcx]
                    _phwm = np.where(
                        _phden > 0,
                        _ph_num / np.maximum(_phden, 1e-30), 0.0)
                    _thr_ph = float(bright_mask_sub_nsigma
                                    if bright_mask_sub_nsigma else 5.0)
                    _dem = _phwm > (_thr_ph * _nz[_phcy, _phcx])
                    if _dem.any():
                        _drows = _phi[_dem]
                        ks1_flux[_drows, :] = 0.0
                        demoted_glob.extend(
                            int(keep_idx[r]) for r in _drows)
                    _phs1 = ks1_flux[_phi, ref_band]
                    _s1err = np.sqrt(np.maximum(kvar[_phi, ref_band], 0.0)) \
                        if kvar is not None else np.zeros(len(_phi))
                    _neg = _phs1 < -3.0 * np.where(_s1err > 0, _s1err, np.inf)
                    logger.info(
                        "grid_gpu[island]: Exp C v2: %d phantom column(s); "
                        "%d demoted by footprint protection; G3 monitor: "
                        "%d (%.1f%%) significantly negative S1 fluxes "
                        "(ref band %s)",
                        len(_phi), int(_dem.sum()), int(_neg.sum()),
                        100.0 * float(_neg.mean()) if len(_phi) else 0.0,
                        bandnames_with_psf[ref_band])
                # Persist the phantom table (positions + S1 ref-band flux +
                # demotion flag): phantom rows are dropped from the output
                # catalog, so attribution analyses need this file.
                if _has_ph and round_idx == 0 and work_dir:
                    try:
                        import pandas as _pd2
                        _phi2 = np.nonzero(kph_pad)[0]
                        _dset = set(demoted_glob)
                        _pd2.DataFrame({
                            "x": kx[_phi2], "y": ky[_phi2],
                            "s1_flux_ref": ks1_flux[_phi2, ref_band],
                            "demoted": [int(keep_idx[r]) in _dset
                                        for r in _phi2],
                        }).to_csv(os.path.join(str(work_dir),
                                               "phantom_columns.csv"),
                                  index=False)
                        logger.info(
                            "grid_gpu[island]: phantom table -> %s",
                            os.path.join(str(work_dir),
                                         "phantom_columns.csv"))
                    except Exception as _e:
                        logger.warning(
                            "phantom_columns.csv write failed: %s", _e)
                _CH = 2048
                # T3 (2026-08-14, development ledger SS14): the render chain
                # is eager (un-jitted vmap -- every primitive is its own
                # kernel launch), so each 2048-source chunk costs a
                # near-fixed 0.13-0.32 s in DISPATCH regardless of chunk
                # size, and both backends compute a full galaxy render per
                # stamp that jnp.where then discards for STAR rows. jitting
                # the chain is ~50x faster but shifts stamps at the ulp
                # level, which the island solver amplifies beyond the
                # run-to-run envelope (sparse p99 0.58 mmag, 8 reliability
                # flips) -- rejected for a perf item. Shipped instead: an
                # EXACT star-only eager path (the same primitive calls the
                # STAR rows already take: evaluate_psf_at + lanczos shift),
                # verified 0-ulp against the full chain's star rows AT THE TIME (T3); since
                # effarc2 K1 rewrote the shared Lanczos taps, neither path is
                # bit-identical to the pre-K1 engine any more -- they remain
                # identical to EACH OTHER, which is what this fallback needs
                # under unchanged chunk membership
                # (t3_bench/bench_star_only_exact.py C1). Batch composition
                # is NOT fp-neutral (C2/C3), so the chunk grid must stay
                # untouched. A chunk falls back to the original full-chain
                # call iff it contains an extended source whose stamp the
                # Exp B numerator below will actually read -- every value
                # ever read from _st is then bit-identical to the pre-T3
                # code on any anchor.
                def _star_stamps_eager(_pjx, _xs32, _ys32):
                    # effarc2 K6 (2026-08-26): the SAME primitive chain the T3
                    # exact star-only path uses, but COMPILED once per n_terms
                    # instead of dispatched primitive by primitive. T3 kept it
                    # eager because jitting perturbs the stamps at the ulp
                    # level and the arc's contract then was a same-lineage
                    # envelope (development ledger SS14-3a); the 2026-08-26
                    # contract is Gate B -- the same-code cold-run
                    # DISTRIBUTION -- so that perturbation is in-envelope, and
                    # K1 already re-expressed the Lanczos taps anyway. Chunk
                    # membership is unchanged (the chunk grid stays the
                    # fp-relevant knob, T3 C2/C3).
                    # .shape is static metadata; np.asarray() here would pull
                    # the whole coefficient cube back from the device.
                    _fn = _star_stamps_jit(int(_pjx.coefs.shape[-1]))
                    return np.asarray(_fn(
                        _jnp.asarray(_pjx.coefs),
                        _jnp.float32(_pjx.x_origin),
                        _jnp.float32(_pjx.x_scale),
                        _jnp.float32(_pjx.y_origin),
                        _jnp.float32(_pjx.y_scale),
                        _jnp.asarray(_xs32), _jnp.asarray(_ys32)))

                _needs_full_src = np.zeros(int(kx.size), bool)
                if _det_by_src is not None:
                    for _gi, _cand in _det_by_src.items():
                        if _cand:
                            _needs_full_src[_gi] = True
                # Render-vs-scatter split (chunk-granular accumulators only --
                # nothing is timed inside the per-source loop itself).
                _t_render = 0.0
                _t_scatter = 0.0
                _n_full_chunks = 0
                _n_star_chunks = 0
                for bi in range(n_bands):
                    _pj = psfs_per_band_objs[bi].to_jax()
                    for s0 in range(0, int(kx.size), _CH):
                        s1e = min(s0 + _CH, int(kx.size))
                        _tr0 = time.perf_counter()
                        if _needs_full_src[s0:s1e].any():
                            if bi == 0:
                                _n_full_chunks += 1
                            _st = np.asarray(render_source_stamps(
                                _pj,
                                _jnp.asarray(kx[s0:s1e].astype(np.float32)),
                                _jnp.asarray(ky[s0:s1e].astype(np.float32)), 0, 0,
                                _jnp.asarray(kst[s0:s1e].astype(np.int32)),
                                _jnp.asarray(_lr[s0:s1e]),
                                _jnp.asarray(_e1[s0:s1e]),
                                _jnp.asarray(_e2[s0:s1e]),
                                sersic_n=_jnp.asarray(ksn[s0:s1e].astype(np.float32)),
                                render_backend=render_backend))
                        else:
                            if bi == 0:
                                _n_star_chunks += 1
                            _st = _star_stamps_eager(
                                _pj, kx[s0:s1e].astype(np.float32),
                                ky[s0:s1e].astype(np.float32))
                        _t_render += time.perf_counter() - _tr0
                        _fb = ks1_flux[s0:s1e, bi]
                        _ts0 = time.perf_counter()
                        for _k in range(s1e - s0):
                            _f = float(_fb[_k])
                            _gk = s0 + _k
                            # Exp B: extended sources are never subtracted,
                            # but their S1 seed-shape model value at nearby
                            # unmatched detections feeds the footprint-
                            # protection test (white-numerator accumulation
                            # with the same iv weighting as _wh).
                            if (_det_by_src is not None and _f != 0.0
                                    and _isext[_gk]):
                                _dl = _det_by_src.get(int(_gk))
                                if _dl:
                                    _ixk = int(_ixt[_gk])
                                    _iyk = int(_iyt[_gk])
                                    for _di in _dl:
                                        _sy = int(_det_pyp[_di]) - _iyk
                                        _sx = int(_det_pxp[_di]) - _ixk
                                        if 0 <= _sy < Tt and 0 <= _sx < Tt:
                                            _det_num[_di] += (
                                                float(iv[bi,
                                                         _det_pyp[_di],
                                                         _det_pxp[_di]])
                                                * _f
                                                * float(_st[_k, _sy, _sx]))
                            # STAR-only: subtract only sources we can model
                            # EXACTLY from the catalog+S1 (PSF, no free shape).
                            # Extended non-members subtracted at their SEED
                            # (compact) shape inject a neg-core/pos-wing artifact
                            # that hurts more than the smooth contamination
                            # (measured: big-galaxy flux worse 6/8) -> leave
                            # them to the joint fit / tier-3 iteration.
                            if _f == 0.0 or _isext[_gk]:
                                continue
                            ix = int(_ixt[_gk]); iy = int(_iyt[_gk])
                            y0c = max(0, -iy); x0c = max(0, -ix)
                            ye = min(Tt, Hp - iy); xe = min(Tt, Wp - ix)
                            if ye <= y0c or xe <= x0c:
                                continue
                            dmb[bi, iy + y0c:iy + ye, ix + x0c:ix + xe] -= \
                                _f * _st[_k, y0c:ye, x0c:xe]
                        _t_scatter += time.perf_counter() - _ts0
                # Exp B footprint protection: drop unmatched detections
                # whose extended-model white value alone clears the
                # detection threshold at their pixel (attributable to a
                # cataloged galaxy's own outskirts -> subtracting them as
                # "stars" corrupts that galaxy; measured full-field).
                _n_prot = 0
                if _det_num is not None:
                    _dcx = np.clip(
                        np.round(_sub_x).astype(np.int64), 0, W - 1)
                    _dcy = np.clip(
                        np.round(_sub_y).astype(np.int64), 0, H - 1)
                    _dden = _ivs[_dcy, _dcx]
                    _dwm = np.where(
                        _dden > 0, _det_num / np.maximum(_dden, 1e-30), 0.0)
                    # per-candidate threshold: each detection is protected
                    # against the significance of the pass that found it
                    _prot = _dwm > (_sub_thr * _nz[_dcy, _dcx])
                    _n_prot = int(_prot.sum())
                    if _n_prot:
                        if _det_rows is not None:
                            _det_rows["protected"][_sub_rows[_prot]] = True
                        _sub_x = _sub_x[~_prot]
                        _sub_y = _sub_y[~_prot]
                        _sub_rows = _sub_rows[~_prot]
                        _sub_thr = _sub_thr[~_prot]
                # subtract-only uncataloged STARS: per-band PSF-forced flux on the
                # now catalog-cleaned dmb (fit-set stars removed above), then
                # subtract. These NEVER enter islanding / the joint fit.
                n_sub = 0
                _sub_flux = None
                if _sub_x is not None and len(_sub_x):
                    n_sub = int(len(_sub_x))
                    _sub_flux = np.full((n_sub, n_bands), np.nan, np.float64)
                    _sxt = np.round(_sub_x).astype(np.int64) - Tt // 2 + PAD
                    _syt = np.round(_sub_y).astype(np.int64) - Tt // 2 + PAD
                    _z = np.zeros(n_sub, np.float32)
                    for bi in range(n_bands):
                        # All rows here are STARs by construction (subtract-only
                        # uncataloged), so the exact star-only path applies
                        # unconditionally: identical to the full-chain call on
                        # this input (it selects raw_psf for every row); both moved
                        # together when effarc2 K1 rewrote the Lanczos taps.
                        _pj = psfs_per_band_objs[bi].to_jax()
                        _stmp = _star_stamps_eager(
                            _pj, _sub_x.astype(np.float32),
                            _sub_y.astype(np.float32))
                        for _k in range(n_sub):
                            ix = int(_sxt[_k]); iy = int(_syt[_k])
                            y0c = max(0, -iy); x0c = max(0, -ix)
                            ye = min(Tt, Hp - iy); xe = min(Tt, Wp - ix)
                            if ye <= y0c or xe <= x0c:
                                continue
                            _s = _stmp[_k, y0c:ye, x0c:xe]
                            _d = dmb[bi, iy + y0c:iy + ye, ix + x0c:ix + xe]
                            _w = iv[bi, iy + y0c:iy + ye, ix + x0c:ix + xe]
                            _dn = float((_w * _s * _s).sum())
                            if _dn <= 0.0:
                                continue
                            _ff = float((_w * _s * _d).sum()) / _dn
                            if _ff <= 0.0:
                                continue
                            _sub_flux[_k, bi] = _ff
                            dmb[bi, iy + y0c:iy + ye, ix + x0c:ix + xe] -= _ff * _s
                logger.info(
                    "grid_gpu[island]: round %d: scene-subtract removed %d STAR "
                    "S1 models + %d subtract-only uncataloged (%d ext left in) in %.1fs "
                    "(render+D2H %.1fs, host-scatter %.1fs; chunks star-only %d / full %d)",
                    round_idx, int((~_isext).sum()), n_sub, int(_isext.sum()),
                    time.perf_counter() - t0ss, _t_render, _t_scatter,
                    _n_star_chunks, _n_full_chunks)
                # Persist the detection table (Exp B attribution: which
                # detections were matched / footprint-protected / actually
                # subtracted, with their per-band forced fluxes).
                if _det_rows is not None and work_dir:
                    try:
                        import pandas as _pd
                        _dfd = _pd.DataFrame(_det_rows)
                        _dfd["subtracted"] = False
                        if n_sub and _sub_flux is not None:
                            _ok = np.isfinite(_sub_flux).any(axis=1)
                            _dfd.loc[_sub_rows[_ok], "subtracted"] = True
                            for _bi2, _bn2 in enumerate(bandnames_with_psf):
                                _col = np.full(len(_dfd), np.nan)
                                _col[_sub_rows] = _sub_flux[:, _bi2]
                                _dfd[f"flux_{_bn2}"] = _col
                        _csvp = os.path.join(
                            str(work_dir),
                            "subtract_detected.csv" if round_idx == 0
                            else f"subtract_detected_r{round_idx}.csv")
                        _dfd.to_csv(_csvp, index=False)
                        logger.info(
                            "grid_gpu[island]: subtract-only candidates: %d "
                            "(%d sep + %d bmask-routed; %d matched to "
                            "catalog, %d footprint-protected near extended, "
                            "%d subtracted) -> %s",
                            len(_dfd), int((_dfd["src"] == "sep").sum()),
                            int((_dfd["src"] == "bmask").sum()),
                            int(_dfd["matched"].sum()), _n_prot,
                            int(_dfd["subtracted"].sum()), _csvp)
                    except Exception as _e:
                        logger.warning(
                            "subtract_detected CSV write failed: %s", _e)
            logger.info(
                "grid_gpu[island]: round %d: padded field data-bg/invvar "
                "stacks built in %.1fs (%.2f GiB shm x2, pad=%d); "
                "sigma_sky=%.4g (median over %d bands)",
                round_idx, time.perf_counter() - t0s,
                dmb.nbytes / 2 ** 30, PAD, sigma_sky, len(sigma_bands))

            # Catalog-driven prior sigma columns, sliced to the fit set
            # (keep_idx) so plan.midx indexes them consistently. Absent
            # (None) -> class_host_arrays treats as all-NaN -> inv_sigma 0.
            # #34: appended DET rows carry NO prior -> pad NaN so midx
            # gathers stay in-bounds (a detected blob is a seed, not belief).
            def _kslice(arr):
                if arr is None:
                    return None
                base = np.asarray(arr)[keep_idx]   # unchanged dtype (off path)
                if n_det:
                    base = np.concatenate(
                        [base, np.full(n_det, np.nan, dtype=base.dtype)])
                return base

            src_kwargs = dict(
                pos_x=kx.astype(np.float32), pos_y=ky.astype(np.float32),
                source_type=kst, logre=klr, ee1=ke1, ee2=ke2, sersic_n=ksn,
                free_n_all=free_n_all,
                # #35 scene-subtract: per-band S1 fluxes for the in-kernel
                # member add-back (class_host_arrays gathers member_s1_flux =
                # s1_flux[midx]). Only consumed when scene_subtract is on; the
                # kernel DCE's the add-back otherwise (byte-identical off).
                s1_flux=ks1_flux if scene_subtract else None,
                pos_max_shift_pix=float(pos_max_shift_pix), pad=PAD,
                pos_err=_kslice(pos_err_prior),
                re_prior_sig=_kslice(re_prior_sig),
                ell_prior_sig=_kslice(ell_prior_sig),
                theta_prior_sig=_kslice(theta_prior_sig),
                sersic_n_prior_sig=_kslice(sersic_n_prior_sig),
            )
            payload = dict(
                data_spec=(dmb_seg.name, (n_bands, Hp, Wp), "float32"),
                invvar_spec=(iv_seg.name, (n_bands, Hp, Wp), "float32"),
                bandnames=list(bandnames_with_psf),
                psfs=psfs_per_band_objs,
                plans=plans,
                island_set=island_set,
                src_kwargs=src_kwargs,
                solver_opts=dict(
                    render_backend=str(render_backend), reg=1e-9,
                    pos_step_max_pix=DEFAULT_POS_STEP_MAX,
                    chi2_ftol_rel=DEFAULT_CHI2_FTOL_REL,
                    step_tol_pix=DEFAULT_STEP_TOL_PIX,
                    un_freeze_steps=DEFAULT_UN_FREEZE_STEPS,
                    star_render_split=star_render_split,
                    stall_count_mode=stall_count_mode,
                    step_mode=step_mode,
                    scene_subtract=scene_subtract,
                    # dlnp campaign: "legacy" (default) = byte-identical
                    # pre-dlnp step programs; the tol enters kernels only
                    # in dlnp mode (the statistical quantum 2*tol).
                    convergence=convergence,
                    plateau_dlnp_tol=plateau_dlnp_tol,
                ),
                pad=PAD,
                field_shape=(n_bands, H, W),
                warm_sersic_table=(str(render_backend) == "fft"
                                   and bool(np.any(kst != 0))),
                compute_param_errors=param_errors,
                compute_source_metrics=source_metrics,
                compute_flux_marg=marginal_flux_err,
            )

            t0f = time.perf_counter()
            logger.info(
                "grid_gpu[island]: round %d: island fit stage starting -- "
                "%d chunk-sharded worker(s) on GPU(s) %s; re-queue budgets: "
                "main=%d hard=%d step(s)%s%s", round_idx,
                len(devices), list(devices), budget_main, budget_hard,
                " (+residual export)" if want_residual else "",
                ("" if convergence == "legacy"
                 else f"; convergence={convergence} "
                      f"(quantum={2.0 * plateau_dlnp_tol:g})"))
            ires, shard_stats, residuals = _run_island_fit_sharded(
                devices=list(devices), ctx=ctx, mem_fraction=mem_fraction,
                jit_cache_dir=jit_cache_dir, worker_log_file=worker_log_file,
                payload=payload, plans=plans, island_set=island_set,
                giant_ids=giant_ids, budget_main=budget_main,
                budget_hard=budget_hard, n_bands=n_bands,
                export_residual=bool(want_residual),
                width_cache_path=(
                    os.path.join(str(work_dir), "island_chunk_widths.json")
                    if work_dir else None))
            t_island = time.perf_counter() - t0f
            busy = shard_stats["per_gpu_busy_s"]
            logger.info(
                "grid_gpu[island]: round %d: island fit stage done in %.1fs "
                "wall -- %d source rows, unconverged %d (%.2f%%), stalled "
                "%d, giant rows %d, free-n rows %d; shard balance: per-GPU "
                "busy %s (total %.1fs, ideal makespan %.1fs, achieved "
                "%.1fs); re-queued %d island(s), hard-pass %.1fs GPU "
                "(%.1f%% of busy)", round_idx,
                t_island, ires.source_idx.size,
                int((~ires.converged).sum()),
                100.0 * float((~ires.converged).mean()) if ires.source_idx.size else 0.0,
                int(ires.stalled.sum()), int(ires.island_giant.sum()),
                int(ires.free_n.sum()), busy, shard_stats["total_busy_s"],
                shard_stats["ideal_makespan_s"], shard_stats["makespan_s"],
                shard_stats["n_requeued_islands"], shard_stats["hard_gpu_s"],
                100.0 * shard_stats["hard_gpu_s"]
                / max(shard_stats["total_busy_s"], 1e-9))
            run_status.note(f"island_round_{round_idx}", {
                "n_islands": int(island_set.n_src.size),
                "n_giant_islands": int(np.unique(
                    ires.island_id[ires.island_giant]).size),
                "source_rows": int(ires.source_idx.size),
                "giant_rows": int(ires.island_giant.sum()),
                "unconverged_rows": int((~ires.converged).sum()),
                "stalled_rows": int(ires.stalled.sum()),
                "wall_s": round(float(t_island), 1)})

            # ---- Item-4 Layer-2 blend-resolve, mode "detect" (R1) -------
            # Runs while dmb/iv (the padded, S1-bg-subtracted, all-STAR-
            # scene-cleaned field stacks) are still alive. For each
            # BLEND-SUSPECT host (per-source rchisq above the flag
            # convention AND fit-vs-seed displacement above the run's
            # median PSF core sigma -- the flag_blend_suspect operating
            # point), reconstruct the local S1-STATE residual (dmb minus
            # every extended source's seed-shape x S1-flux model; stars
            # are already scene-subtracted) and SEP-detect point-source
            # candidates at br_nsigma x the local white noise. Candidate
            # gates (all derived): separation from the host > PSF core
            # sigma (an on-core star column is flux-degenerate with the
            # host -- measured), and within the PSF stamp radius.
            # Disclosure only: {work_dir}/blend_candidates.csv; the fit,
            # meta, and pass-2 tasks are untouched.
            if br_on and ires.source_idx.size:
                try:
                    t0br = time.perf_counter()
                    import sep as _sep3
                    from ...sep_limits import apply_sep_limits
                    apply_sep_limits(_sep3)
                    import jax.numpy as _jnp3
                    from .forced_phot_sparse import (
                        render_source_stamps as _render3)
                    _em_br = (gpu_cfg.get("error_model", {}) or {})
                    # same convention as flag_high_rchisq; the shipped
                    # flag threshold when set, else the documented
                    # reduced-chi2 misfit convention (3).
                    _rct = _em_br.get("rchisq_flag_thresh", None)
                    _rct = float(_rct) if _rct else 3.0
                    _rcv = np.asarray(ires.rchisq, np.float64)
                    if not np.any(np.isfinite(_rcv)):
                        logger.warning(
                            "blend_resolve: per-source rchisq unavailable "
                            "(error_model.source_metrics_enabled off?) -- "
                            "detector skipped")
                        raise StopIteration
                    # PSF core sigma: identical recipe to the
                    # flag_blend_suspect emission (half-max isophotal
                    # area FWHM, wing-robust; NOT the wing-dominated
                    # stamp second moment), median over bands.
                    _sigb = []
                    for _bi3 in range(n_bands):
                        _phi3 = np.clip(np.asarray(evaluate_psf_at_np(
                            psfs_per_band_objs[_bi3], W * 0.5, H * 0.5),
                            dtype=float), 0.0, None)
                        _pk3 = float(_phi3.max()) if _phi3.size else 0.0
                        if _pk3 <= 0 or not np.isfinite(_pk3):
                            continue
                        _nh3 = int(np.count_nonzero(_phi3 > 0.5 * _pk3))
                        if _nh3 > 0:
                            _sigb.append(
                                2.0 * float(np.sqrt(_nh3 / np.pi)) / 2.355)
                    _sig_core = float(np.median(_sigb)) if _sigb else np.nan
                    if not (np.isfinite(_sig_core) and _sig_core > 0):
                        logger.warning(
                            "blend_resolve: PSF core sigma unavailable -- "
                            "detector skipped")
                        raise StopIteration
                    _rowsb = np.asarray(ires.source_idx, np.int64)
                    _shiftb = np.hypot(
                        np.asarray(ires.pos_x, float) - kx[_rowsb],
                        np.asarray(ires.pos_y, float) - ky[_rowsb])
                    _glb = fit_to_global[_rowsb]
                    _realb = _glb >= 0
                    if phantom_mask is not None:
                        _pmv = np.zeros(_realb.shape, bool)
                        _pmv[_realb] = phantom_mask[_glb[_realb]]
                        _realb &= ~_pmv
                    _sus = (_realb & np.isfinite(_rcv) & (_rcv > _rct)
                            & np.isfinite(_shiftb)
                            & (_shiftb > _sig_core))
                    _T3 = int(native)
                    _hB3 = _T3 // 2
                    _cand_out = []
                    _br_cut = {}   # mode "resolve": per-host S1 cutouts
                    _n_with = 0
                    for _si in np.nonzero(_sus)[0]:
                        _fr = int(_rowsb[_si])
                        _cx3 = float(kx[_fr]); _cy3 = float(ky[_fr])
                        _icx = int(round(_cx3)); _icy = int(round(_cy3))
                        _x0c = _icx - _hB3 + PAD; _y0c = _icy - _hB3 + PAD
                        _x1c = _x0c + _T3; _y1c = _y0c + _T3
                        if (_x0c < 0 or _y0c < 0 or _x1c > Wp
                                or _y1c > Hp):
                            continue
                        _res3 = np.array(dmb[:, _y0c:_y1c, _x0c:_x1c],
                                         dtype=np.float64, copy=True)
                        _wv3 = np.asarray(iv[:, _y0c:_y1c, _x0c:_x1c],
                                          dtype=np.float64)
                        # subtract extended sources' seed-shape S1 models
                        # (the host itself included -- the S1 state)
                        _near = np.nonzero(
                            _isext
                            & (np.abs(kx - _cx3) < _T3)
                            & (np.abs(ky - _cy3) < _T3))[0]
                        for _bi3 in range(n_bands):
                            if _near.size == 0:
                                break
                            _pj3 = psfs_per_band_objs[_bi3].to_jax()
                            _stm3 = np.asarray(_render3(
                                _pj3,
                                _jnp3.asarray(kx[_near].astype(np.float32)),
                                _jnp3.asarray(ky[_near].astype(np.float32)),
                                0, 0,
                                _jnp3.asarray(kst[_near].astype(np.int32)),
                                _jnp3.asarray(_lr[_near]),
                                _jnp3.asarray(_e1[_near]),
                                _jnp3.asarray(_e2[_near]),
                                sersic_n=_jnp3.asarray(
                                    ksn[_near].astype(np.float32)),
                                render_backend=render_backend))
                            for _k3, _j3 in enumerate(_near):
                                _f3 = float(ks1_flux[_j3, _bi3])
                                if _f3 == 0.0 or not np.isfinite(_f3):
                                    continue
                                _jx = (int(round(kx[_j3])) - _T3 // 2
                                       + PAD - _x0c)
                                _jy = (int(round(ky[_j3])) - _T3 // 2
                                       + PAD - _y0c)
                                _dy0 = max(0, _jy)
                                _dy1 = min(_T3, _jy + _T3)
                                _dx0 = max(0, _jx)
                                _dx1 = min(_T3, _jx + _T3)
                                if _dy1 <= _dy0 or _dx1 <= _dx0:
                                    continue
                                _res3[_bi3, _dy0:_dy1, _dx0:_dx1] -= (
                                    _f3 * _stm3[_k3, _dy0 - _jy:_dy1 - _jy,
                                                _dx0 - _jx:_dx1 - _jx])
                        _den3 = _wv3.sum(0)
                        _bad3 = _den3 <= 0
                        _dsafe = np.where(_bad3, 1.0, _den3)
                        _wh3 = ((_wv3 * _res3).sum(0) / _dsafe)
                        _nz3 = np.where(
                            _bad3, 1e30,
                            1.0 / np.sqrt(_dsafe)).astype(np.float32)
                        try:
                            _o3 = _sep3.extract(
                                np.ascontiguousarray(
                                    _wh3.astype(np.float32)),
                                thresh=br_nsigma, err=_nz3, mask=_bad3,
                                minarea=br_minarea, deblend_nthresh=32,
                                deblend_cont=0.005, clean=True,
                                clean_param=1.0)
                        except Exception:
                            _o3 = None
                        _hx3 = _cx3 - _icx + _hB3
                        _hy3 = _cy3 - _icy + _hB3
                        _kept = []
                        if _o3 is not None and len(_o3):
                            _dsep = np.hypot(_o3["x"] - _hx3,
                                             _o3["y"] - _hy3)
                            _pxi = np.clip(np.round(_o3["x"]).astype(int),
                                           0, _T3 - 1)
                            _pyi = np.clip(np.round(_o3["y"]).astype(int),
                                           0, _T3 - 1)
                            _psig = _o3["peak"] / np.maximum(
                                _nz3[_pyi, _pxi], 1e-30)
                            # NOTE a "must be off every other modeled
                            # source" gate was tried and MEASURED-
                            # NEGATIVE (expanded oracle, paired 63
                            # hosts: 7 better without / 0 better with):
                            # peaks at phantom positions are usually
                            # the phantom's genuine UNDER-subtraction,
                            # and resolving them (joint solve + refit +
                            # guard) tops the correction up. Reverted.
                            _gok = ((_dsep > _sig_core)
                                    & (_dsep <= _T3 / 2.0))
                            _kept = list(np.nonzero(_gok)[0])
                        # SUB-PIXEL POSITION REFINEMENT (cutout frame).
                        # SEP's x/y is the isophotal-area BARYCENTER, which
                        # a star on the host's residual gradient pulls ~1 px
                        # toward the host (measured: caps the pass-2 star
                        # flux split at 0.82x, leaving residual host
                        # contamination). The flux PEAK is not pulled: refine
                        # to the peak pixel + a 3-point parabolic sub-pixel
                        # offset on the same white-stack S1-residual _wh3
                        # (SExtractor's classic peak interpolation; the vertex
                        # of a quadratic through 3 samples -- derived, no
                        # tuned constants). Fall back to the barycenter if the
                        # local curvature is not concave-down (flat / noise)
                        # or the peak sits on the cutout edge.
                        _refx = {}
                        _refy = {}
                        _names3 = (_o3.dtype.names
                                   if _o3 is not None else None)
                        _has_pk = bool(_names3 and "xpeak" in _names3
                                       and "ypeak" in _names3)
                        for _ci in _kept:
                            _xb = float(_o3["x"][_ci])
                            _yb = float(_o3["y"][_ci])
                            _refx[_ci] = _xb
                            _refy[_ci] = _yb
                            if not _has_pk:
                                continue
                            _xp = int(_o3["xpeak"][_ci])
                            _yp = int(_o3["ypeak"][_ci])
                            if not (1 <= _xp < _T3 - 1
                                    and 1 <= _yp < _T3 - 1):
                                continue
                            _cpk = float(_wh3[_yp, _xp])
                            _lx = float(_wh3[_yp, _xp - 1])
                            _rx = float(_wh3[_yp, _xp + 1])
                            _ly = float(_wh3[_yp - 1, _xp])
                            _ry = float(_wh3[_yp + 1, _xp])
                            _dxd = _lx - 2.0 * _cpk + _rx
                            _dyd = _ly - 2.0 * _cpk + _ry
                            _dx = (0.5 * (_lx - _rx) / _dxd
                                   if _dxd < 0 else 0.0)
                            _dy = (0.5 * (_ly - _ry) / _dyd
                                   if _dyd < 0 else 0.0)
                            if abs(_dx) <= 1.0 and abs(_dy) <= 1.0:
                                _refx[_ci] = _xp + _dx
                                _refy[_ci] = _yp + _dy
                        if not _kept:
                            _cand_out.append(dict(
                                host_global=int(_glb[_si]),
                                host_x=_cx3, host_y=_cy3,
                                host_rchisq=float(_rcv[_si]),
                                host_seed_shift_px=float(_shiftb[_si]),
                                cand_x=np.nan, cand_y=np.nan,
                                cand_sep_px=np.nan, cand_peak_sig=np.nan,
                                n_cand=0, best=False))
                            continue
                        _n_with += 1
                        _best = _kept[int(np.argmax(_psig[_kept]))]
                        # Candidate fluxes, per band. ONE batched render
                        # per band for [all kept candidates (as STARs) +
                        # the host at its seed shape]. Every candidate
                        # gets the plain matched-filter amplitude on the
                        # S1-state residual; the BEST candidate is then
                        # UPGRADED to the exact 2x2 JOINT [host, star]
                        # amplitude solve on the host-restored local data
                        # -- the single-pass MF systematically under-
                        # recovers (0.46-0.78x measured) because the part
                        # of the star the host's S1 flux absorbed lives
                        # in the HOST MODEL, not the residual; the joint
                        # solve reapportions it in closed form.
                        _nk3 = len(_kept)
                        _bx3 = np.concatenate([
                            np.float32([_refx[_ci]
                                        + (_icx - _hB3)
                                        for _ci in _kept]),
                            np.float32([_cx3])])
                        _by3 = np.concatenate([
                            np.float32([_refy[_ci]
                                        + (_icy - _hB3)
                                        for _ci in _kept]),
                            np.float32([_cy3])])
                        _bt3 = np.concatenate([
                            np.zeros(_nk3, np.int32),
                            np.int32([kst[_fr]])])
                        _blr = np.concatenate([
                            np.zeros(_nk3, np.float32),
                            np.float32([_lr[_fr] if _isext[_fr]
                                        else 0.0])])
                        _be1 = np.concatenate([
                            np.ones(_nk3, np.float32),
                            np.float32([_e1[_fr] if _isext[_fr]
                                        else 1.0])])
                        _be2 = np.concatenate([
                            np.zeros(_nk3, np.float32),
                            np.float32([_e2[_fr] if _isext[_fr]
                                        else 0.0])])
                        _bsn = np.concatenate([
                            np.ones(_nk3, np.float32),
                            np.float32([ksn[_fr]])])
                        _mf_flux = np.full((_nk3, n_bands), np.nan)
                        _joint_flux = np.full(n_bands, np.nan)

                        def _win3(_xc, _yc, _ts):
                            _jx = int(round(_xc)) - _ts // 2 + PAD - _x0c
                            _jy = int(round(_yc)) - _ts // 2 + PAD - _y0c
                            _dy0 = max(0, _jy); _dy1 = min(_T3, _jy + _ts)
                            _dx0 = max(0, _jx); _dx1 = min(_T3, _jx + _ts)
                            return _jx, _jy, _dy0, _dy1, _dx0, _dx1

                        for _bi3 in range(n_bands):
                            _pj3 = psfs_per_band_objs[_bi3].to_jax()
                            _stb = np.asarray(_render3(
                                _pj3, _jnp3.asarray(_bx3),
                                _jnp3.asarray(_by3), 0, 0,
                                _jnp3.asarray(_bt3),
                                _jnp3.asarray(_blr),
                                _jnp3.asarray(_be1),
                                _jnp3.asarray(_be2),
                                sersic_n=_jnp3.asarray(_bsn),
                                render_backend=render_backend))
                            _ts = _stb.shape[1]
                            # host model restored: d' = residual + f_S1*h
                            _dprime = _res3[_bi3].copy()
                            _fh = float(ks1_flux[_fr, _bi3])
                            _hst = np.zeros_like(_dprime)
                            _jx, _jy, _dy0, _dy1, _dx0, _dx1 = _win3(
                                _cx3, _cy3, _ts)
                            if _dy1 > _dy0 and _dx1 > _dx0:
                                _hst[_dy0:_dy1, _dx0:_dx1] = _stb[
                                    _nk3, _dy0 - _jy:_dy1 - _jy,
                                    _dx0 - _jx:_dx1 - _jx]
                            if np.isfinite(_fh) and _fh != 0.0:
                                _dprime += _fh * _hst
                            _ww = _wv3[_bi3]
                            for _q3, _ci in enumerate(_kept):
                                _jx, _jy, _dy0, _dy1, _dx0, _dx1 = _win3(
                                    _bx3[_q3], _by3[_q3], _ts)
                                if _dy1 <= _dy0 or _dx1 <= _dx0:
                                    continue
                                _ss = np.zeros_like(_dprime)
                                _ss[_dy0:_dy1, _dx0:_dx1] = _stb[
                                    _q3, _dy0 - _jy:_dy1 - _jy,
                                    _dx0 - _jx:_dx1 - _jx]
                                _dn3 = float((_ww * _ss * _ss).sum())
                                if _dn3 > 0:
                                    _mf_flux[_q3, _bi3] = float(
                                        (_ww * _ss * _res3[_bi3]).sum()
                                    ) / _dn3
                                if _ci == _best:
                                    # exact 2x2 joint [host, star] solve
                                    _hh = float((_ww * _hst * _hst).sum())
                                    _hs = float((_ww * _hst * _ss).sum())
                                    _sw = float((_ww * _ss * _ss).sum())
                                    _bh = float((_ww * _hst
                                                 * _dprime).sum())
                                    _bs = float((_ww * _ss
                                                 * _dprime).sum())
                                    _det3 = _hh * _sw - _hs * _hs
                                    if (_det3 > 0 and _hh > 0
                                            and _sw > 0
                                            and _det3 > 1e-6 * _hh * _sw):
                                        _joint_flux[_bi3] = (
                                            (_hh * _bs - _hs * _bh)
                                            / _det3)
                        for _q3, _ci in enumerate(_kept):
                            _cwx = float(_bx3[_q3]); _cwy = float(_by3[_q3])
                            _isb = bool(_ci == _best)
                            _row = dict(
                                host_global=int(_glb[_si]),
                                host_x=_cx3, host_y=_cy3,
                                host_rchisq=float(_rcv[_si]),
                                host_seed_shift_px=float(_shiftb[_si]),
                                cand_x=_cwx, cand_y=_cwy,
                                cand_sep_px=float(_dsep[_ci]),
                                cand_peak_sig=float(_psig[_ci]),
                                n_cand=len(_kept),
                                best=_isb,
                                flux_est=("joint2" if _isb else "mf"))
                            for _bi3 in range(n_bands):
                                _fv = (_joint_flux[_bi3] if _isb
                                       else _mf_flux[_q3, _bi3])
                                if _isb and not np.isfinite(_fv):
                                    _fv = _mf_flux[_q3, _bi3]  # fallback
                                _row[f"flux_{bandnames_with_psf[_bi3]}"] \
                                    = _fv
                            _cand_out.append(_row)
                        if br_mode == "resolve":
                            # stash the S1-state cutout for the fixed-
                            # point iterate (rejected-island flux
                            # re-solve against the refit host template)
                            _br_cut[_fr] = dict(
                                res=_res3.copy(), x0=_x0c, y0=_y0c)
                    if work_dir is not None:
                        try:
                            import pandas as _pd3
                            _pd3.DataFrame(_cand_out).to_csv(
                                os.path.join(str(work_dir),
                                             "blend_candidates.csv"),
                                index=False)
                        except Exception as _e3:
                            logger.warning(
                                "blend_candidates.csv write failed: %s",
                                _e3)
                    logger.info(
                        "grid_gpu[island]: blend_resolve[detect]: %d "
                        "suspect(s) (rchisq > %.2f AND seed shift > "
                        "psf core sigma %.2f px), %d with >=%.0f-sigma "
                        "candidate(s) in %.1fs -> blend_candidates.csv",
                        int(_sus.sum()), _rct, _sig_core, _n_with,
                        br_nsigma, time.perf_counter() - t0br)

                    # ---- mode "resolve" (R2): subtract each host's BEST
                    # candidate star from dmb at its matched-filter flux
                    # and REFIT only the affected islands from seeds (a
                    # fresh mini shard round on the cleaned data). The
                    # refit chi2 on (data - star) is mathematically the
                    # chi2 of the two-component (host + fixed star) model
                    # on the ORIGINAL data, so the per-island acceptance
                    # guard "refit chi2 <= original chi2" is a like-for-
                    # like nested-model comparison (protects against
                    # optimizer artifacts; the JUNK gate is the detection
                    # significance upstream -- BIC/dchi2 measured-FAIL at
                    # that job). Accepted islands' rows are spliced into
                    # ires; accepted stars are handed to the pass-2
                    # rebuild for a sentinel flux column (without it,
                    # pass-2 re-absorbs the star into the host's flux).
                    if br_mode == "resolve" and _n_with:
                        t0r2 = time.perf_counter()
                        _best_rows = [
                            r for r in _cand_out
                            if r.get("best") and np.isfinite(
                                r.get("cand_x", np.nan))]
                        # host fit-row + island lookups
                        _row_of = {int(_rowsb[_si]): _si
                                   for _si in range(_rowsb.size)}
                        _src2row = {int(s): i for i, s in
                                    enumerate(_rowsb)}
                        _gl2fit = {int(_glb[_si]): int(_rowsb[_si])
                                   for _si in range(_rowsb.size)
                                   if _glb[_si] >= 0}
                        _isl_of = {}
                        _acc_stars = []
                        for _r2 in _best_rows:
                            _hf = _gl2fit.get(int(_r2["host_global"]))
                            if _hf is None or _hf not in _src2row:
                                continue
                            _ri = _src2row[_hf]
                            if bool(ires.island_giant[_ri]):
                                continue   # giants: no island fit, skip
                            _r2["_fit_row"] = _hf
                            _r2["_isl"] = int(ires.island_id[_ri])
                            _r2["_chi2_orig"] = float(ires.chi2[_ri])
                            _isl_of.setdefault(_r2["_isl"], []).append(_r2)
                        if _isl_of:
                            # subtract every candidate star from dmb
                            _z2 = np.zeros(1, np.float32)
                            for _r2s in _isl_of.values():
                                for _r2 in _r2s:
                                    for _bi3 in range(n_bands):
                                        _fb2 = _r2.get(
                                            "flux_"
                                            f"{bandnames_with_psf[_bi3]}",
                                            np.nan)
                                        if (not np.isfinite(_fb2)
                                                or _fb2 <= 0):
                                            continue
                                        _pj3 = (psfs_per_band_objs[_bi3]
                                                .to_jax())
                                        _s3 = np.asarray(_render3(
                                            _pj3,
                                            _jnp3.asarray(np.float32(
                                                [_r2["cand_x"]])),
                                            _jnp3.asarray(np.float32(
                                                [_r2["cand_y"]])),
                                            0, 0,
                                            _jnp3.asarray(
                                                np.zeros(1, np.int32)),
                                            _jnp3.asarray(_z2),
                                            _jnp3.asarray(_z2 + 1.0),
                                            _jnp3.asarray(_z2),
                                            sersic_n=_jnp3.asarray(
                                                _z2 + 1.0),
                                            render_backend=render_backend
                                        ))[0]
                                        _ts = _s3.shape[0]
                                        _jx = (int(round(_r2["cand_x"]))
                                               - _ts // 2 + PAD)
                                        _jy = (int(round(_r2["cand_y"]))
                                               - _ts // 2 + PAD)
                                        _y0s = max(0, _jy)
                                        _y1s = min(Hp, _jy + _ts)
                                        _x0s = max(0, _jx)
                                        _x1s = min(Wp, _jx + _ts)
                                        if _y1s <= _y0s or _x1s <= _x0s:
                                            continue
                                        dmb[_bi3, _y0s:_y1s,
                                            _x0s:_x1s] -= (
                                            np.float32(_fb2)
                                            * _s3[_y0s - _jy:_y1s - _jy,
                                                  _x0s - _jx:_x1s - _jx])
                            # subset IslandSet refit + guard machinery
                            from .islands import IslandSet as _ISet
                            _SPLICE = (
                                "pos_x", "pos_y", "logre", "ee1", "ee2",
                                "sersic_n", "fluxes", "flux_err", "chi2",
                                "chi2_nu", "n_steps", "converged",
                                "stalled", "diverged", "railed",
                                "free_n", "re_upper_pix", "pos_x_err",
                                "pos_y_err", "re_err", "ell_err",
                                "theta_err", "sersic_n_err",
                                "pos_x_prior_frac", "pos_y_prior_frac",
                                "re_prior_frac", "ell_prior_frac",
                                "theta_prior_frac",
                                "sersic_n_prior_frac", "rchisq",
                                "fracflux", "fracin", "fracmasked",
                                "flux_win", "flux_marg_var")
                            _all_src = {int(s): i for i, s in
                                        enumerate(ires.source_idx)}

                            def _refit_subset(_ids):
                                _nm = [island_set.members[_k]
                                       for _k in _ids]
                                _lb = np.full(island_set.n_sources, -1,
                                              np.int32)
                                for _nk, _mem in enumerate(_nm):
                                    _lb[_mem] = _nk
                                _ss2 = _ISet(
                                    n_sources=island_set.n_sources,
                                    n_islands=len(_ids), labels=_lb,
                                    radius=island_set.radius,
                                    members=_nm,
                                    n_src=island_set.n_src[_ids],
                                    bbox=island_set.bbox[_ids],
                                    r_psf=island_set.r_psf)
                                _bk = bucket_islands(
                                    _ss2, ISLAND_BBOX_CLASSES,
                                    ISLAND_NSRC_CLASSES)
                                _pl, _ = plan_island_classes(
                                    _ss2, _bk, kst, klr, ksn, native,
                                    joint_nsrc_cap=joint_nsrc_cap,
                                    log_fn=lambda m: logger.info(
                                        "blend_resolve refit: %s", m))
                                _pay = dict(payload)
                                _pay["plans"] = _pl
                                _pay["island_set"] = _ss2
                                _ir, _, _ = _run_island_fit_sharded(
                                    devices=list(devices), ctx=ctx,
                                    mem_fraction=mem_fraction,
                                    jit_cache_dir=jit_cache_dir,
                                    worker_log_file=worker_log_file,
                                    payload=_pay, plans=_pl,
                                    island_set=_ss2,
                                    giant_ids=np.zeros(0, np.int64),
                                    budget_main=budget_main,
                                    budget_hard=budget_hard,
                                    n_bands=n_bands,
                                    export_residual=False)
                                return _ir

                            def _guard_and_splice(_ids, _ir2):
                                # ACCEPTANCE = per-HOST own-profile
                                # rchisq not-worse (refit vs original).
                                # The island-level chi2 has ~1e5 dof
                                # shared with every member, so the
                                # blend's own 1-2% margin drowns in the
                                # joint refit's fp-level member
                                # variations (measured: a 0.92x-truth
                                # star subtraction got REJECTED by a
                                # +2% island-chi2 wiggle while the host
                                # itself improved). rchisq is the
                                # suspect's own selection metric --
                                # per-source, already computed by the
                                # refit round's source_metrics pass.
                                # Island chi2 kept as a ledger column.
                                # Junk defense stays the 5-sigma
                                # candidate gate upstream.
                                _s2n = {int(s): i for i, s in
                                        enumerate(_ir2.source_idx)}
                                _na = 0
                                _rej = []
                                for _nk, _ok in enumerate(_ids):
                                    _m2 = _ir2.island_id == _nk
                                    if not np.any(_m2):
                                        _rej.append(_ok)
                                        continue
                                    _c2 = float(np.max(_ir2.chi2[_m2]))
                                    _accs_h = []
                                    for _r2 in _isl_of[_ok]:
                                        _ri2 = _s2n.get(
                                            int(_r2["_fit_row"]))
                                        _rc_new = (float(
                                            _ir2.rchisq[_ri2])
                                            if _ri2 is not None
                                            else np.nan)
                                        _rc_old = float(_rcv[
                                            _row_of[_r2["_fit_row"]]])
                                        _r2["rchisq_refit"] = _rc_new
                                        _r2["rchisq_orig"] = _rc_old
                                        _accs_h.append(
                                            np.isfinite(_rc_new)
                                            and _rc_new <= _rc_old)
                                    _acc = bool(_accs_h) and all(_accs_h)
                                    for _r2 in _isl_of[_ok]:
                                        _r2["chi2_refit"] = _c2
                                        _r2["accepted"] = bool(_acc)
                                    if not _acc:
                                        _rej.append(_ok)
                                        continue
                                    _na += 1
                                    _sidx2 = np.asarray(
                                        _ir2.source_idx[_m2], np.int64)
                                    _dst2 = np.full(_sidx2.size, -1,
                                                    np.int64)
                                    for _q, _s2 in enumerate(_sidx2):
                                        _dst2[_q] = _all_src.get(
                                            int(_s2), -1)
                                    _okm = _dst2 >= 0
                                    for _fn in _SPLICE:
                                        _a1 = getattr(ires, _fn)
                                        _a2 = getattr(_ir2, _fn)
                                        if _a1 is None or _a2 is None:
                                            continue
                                        _a1[_dst2[_okm]] = _a2[
                                            np.nonzero(_m2)[0][_okm]]
                                    for _r2 in _isl_of[_ok]:
                                        _acc_stars.append(_r2)
                                return _na, _rej

                            _sub_ids = sorted(_isl_of.keys())
                            ires2 = _refit_subset(_sub_ids)
                            _n_acc, _rej1 = _guard_and_splice(
                                _sub_ids, ires2)

                            # FIXED-POINT ITERATE (exactly one pass) for
                            # rejected islands. The worst blend geometry
                            # (star inside the host's Re, star >> host)
                            # keeps ~8% of the star in the host's own-
                            # profile window even at a 0.92x-truth
                            # subtraction -- a large lump RELATIVE TO THE
                            # HOST, so the refit rchisq honestly worsens
                            # and the guard rejects. One fixed-point step
                            # closes it: re-solve the 2x2 [host, star]
                            # amplitudes against the REFIT host template
                            # (better geometry than the seed), adjust the
                            # dmb subtraction by the flux delta, refit
                            # those islands once more, re-apply the same
                            # guard. Bounded (single step), derived (same
                            # estimator, better template), and touches
                            # only the already-rejected islands.
                            _n_acc2 = 0
                            _it_ids = []
                            if _rej1:
                                _s2n1 = {int(s): i for i, s in
                                         enumerate(ires2.source_idx)}
                                _upd_ids = []
                                for _ok in _rej1:
                                    _did = False
                                    for _r2 in _isl_of[_ok]:
                                        _fr2 = int(_r2["_fit_row"])
                                        _cd = _br_cut.get(_fr2)
                                        _ri1 = _s2n1.get(_fr2)
                                        if _cd is None or _ri1 is None:
                                            continue
                                        _x0c2 = _cd["x0"]
                                        _y0c2 = _cd["y0"]
                                        _resC = _cd["res"]
                                        _wvC = np.asarray(
                                            iv[:, _y0c2:_y0c2 + _T3,
                                               _x0c2:_x0c2 + _T3],
                                            dtype=np.float64)
                                        _hx2 = float(ires2.pos_x[_ri1])
                                        _hy2 = float(ires2.pos_y[_ri1])
                                        _px3 = np.float32(
                                            [_r2["host_x"], _hx2,
                                             _r2["cand_x"]])
                                        _py3 = np.float32(
                                            [_r2["host_y"], _hy2,
                                             _r2["cand_y"]])
                                        _tt3 = np.int32(
                                            [kst[_fr2], kst[_fr2], 0])
                                        _sl3 = np.float32(
                                            [_lr[_fr2],
                                             float(ires2.logre[_ri1]),
                                             0.0])
                                        _se13 = np.float32(
                                            [_e1[_fr2],
                                             float(ires2.ee1[_ri1]),
                                             1.0])
                                        _se23 = np.float32(
                                            [_e2[_fr2],
                                             float(ires2.ee2[_ri1]),
                                             0.0])
                                        _sn3 = np.float32(
                                            [ksn[_fr2],
                                             float(ires2.sersic_n[
                                                 _ri1]), 1.0])
                                        for _bi3 in range(n_bands):
                                            _pj3 = psfs_per_band_objs[
                                                _bi3].to_jax()
                                            _st3 = np.asarray(_render3(
                                                _pj3,
                                                _jnp3.asarray(_px3),
                                                _jnp3.asarray(_py3),
                                                0, 0,
                                                _jnp3.asarray(_tt3),
                                                _jnp3.asarray(_sl3),
                                                _jnp3.asarray(_se13),
                                                _jnp3.asarray(_se23),
                                                sersic_n=_jnp3.asarray(
                                                    _sn3),
                                                render_backend=(
                                                    render_backend)))
                                            _ts = _st3.shape[1]

                                            def _put(_q5, _xc, _yc):
                                                _arr = np.zeros(
                                                    (_T3, _T3),
                                                    np.float64)
                                                _jx = (int(round(_xc))
                                                       - _ts // 2 + PAD
                                                       - _x0c2)
                                                _jy = (int(round(_yc))
                                                       - _ts // 2 + PAD
                                                       - _y0c2)
                                                _dy0 = max(0, _jy)
                                                _dy1 = min(
                                                    _T3, _jy + _ts)
                                                _dx0 = max(0, _jx)
                                                _dx1 = min(
                                                    _T3, _jx + _ts)
                                                if (_dy1 > _dy0
                                                        and _dx1 > _dx0):
                                                    _arr[_dy0:_dy1,
                                                         _dx0:_dx1] = \
                                                        _st3[_q5,
                                                             _dy0 - _jy:
                                                             _dy1 - _jy,
                                                             _dx0 - _jx:
                                                             _dx1 - _jx]
                                                return _arr
                                            _hs0 = _put(
                                                0, _r2["host_x"],
                                                _r2["host_y"])
                                            _hs2 = _put(1, _hx2, _hy2)
                                            _ss2 = _put(
                                                2, _r2["cand_x"],
                                                _r2["cand_y"])
                                            _fh0 = float(
                                                ks1_flux[_fr2, _bi3])
                                            _dp2 = _resC[_bi3].copy()
                                            if (np.isfinite(_fh0)
                                                    and _fh0 != 0.0):
                                                _dp2 += _fh0 * _hs0
                                            _ww2 = _wvC[_bi3]
                                            _hh = float((_ww2 * _hs2
                                                         * _hs2).sum())
                                            _hs_ = float((_ww2 * _hs2
                                                          * _ss2).sum())
                                            _sw = float((_ww2 * _ss2
                                                         * _ss2).sum())
                                            _bh = float((_ww2 * _hs2
                                                         * _dp2).sum())
                                            _bs = float((_ww2 * _ss2
                                                         * _dp2).sum())
                                            _dt = _hh * _sw - _hs_ ** 2
                                            if not (_dt > 1e-6 * _hh
                                                    * _sw and _sw > 0):
                                                continue
                                            _fs_new = (
                                                (_hh * _bs - _hs_ * _bh)
                                                / _dt)
                                            _bn5 = bandnames_with_psf[
                                                _bi3]
                                            _fold = _r2.get(
                                                f"flux_{_bn5}", np.nan)
                                            if not np.isfinite(_fold):
                                                _fold = 0.0
                                            _dfl = _fs_new - _fold
                                            if _dfl != 0.0:
                                                _jx = (int(round(
                                                    _r2["cand_x"]))
                                                    - _ts // 2 + PAD)
                                                _jy = (int(round(
                                                    _r2["cand_y"]))
                                                    - _ts // 2 + PAD)
                                                _y0s = max(0, _jy)
                                                _y1s = min(
                                                    Hp, _jy + _ts)
                                                _x0s = max(0, _jx)
                                                _x1s = min(
                                                    Wp, _jx + _ts)
                                                if (_y1s > _y0s
                                                        and _x1s > _x0s):
                                                    dmb[_bi3,
                                                        _y0s:_y1s,
                                                        _x0s:_x1s] -= (
                                                        np.float32(_dfl)
                                                        * _st3[
                                                            2,
                                                            _y0s - _jy:
                                                            _y1s - _jy,
                                                            _x0s - _jx:
                                                            _x1s - _jx])
                                            _r2[f"flux_{_bn5}"] = \
                                                _fs_new
                                            _did = True
                                        if _did:
                                            _r2["iterated"] = True
                                    if _did:
                                        _upd_ids.append(_ok)
                                if _upd_ids:
                                    _it_ids = _upd_ids
                                    ires3 = _refit_subset(_upd_ids)
                                    _n_acc2, _ = _guard_and_splice(
                                        _upd_ids, ires3)
                            for _r2s in _isl_of.values():
                                for _r2 in _r2s:
                                    br_resolved.append(_r2)
                            logger.info(
                                "grid_gpu[island]: blend_resolve"
                                "[resolve]: %d island(s) refit, %d "
                                "accepted first-pass + %d after the "
                                "fixed-point iterate (%d iterated); "
                                "%d star column(s) -> pass-2, in %.1fs",
                                len(_sub_ids), _n_acc, _n_acc2,
                                len(_it_ids), len(_acc_stars),
                                time.perf_counter() - t0r2)
                except StopIteration:
                    pass
                except Exception as _e3:  # never poison the run
                    logger.warning(
                        "blend_resolve detector failed (fit unaffected): "
                        "%s", str(_e3)[:200])
        finally:
            for seg in shm_handles:
                try:
                    seg.close()
                    seg.unlink()
                except Exception:  # noqa: BLE001
                    pass
        return ires, shard_stats, island_set, t_island, residuals

    try:
        n_det = 0
        fit_to_global = keep_idx           # fit-set row -> global (-1 = DET)
        br_resolved: list[dict] = []       # blend_resolve R2: candidate ledger
        #   (accepted rows become pass-2 sentinel star columns + CSV)
        # ---- bright_mask DET-routing ("model the wings"): route the
        #      bright_mask stage's OWN unmatched detections into the fit set as
        #      compact n-frozen DET (soaked, unemitted) BEFORE round 0, so the
        #      previously-masked neighbours are fit jointly. Same _append_det
        #      machinery as #34, but seeded from the clean pre-fit white-stack
        #      detections (no residual chicken-and-egg). ----
        if bright_mask_det is not None and len(bright_mask_det) > 0:
            bmd = np.asarray(bright_mask_det, dtype=np.float64)
            logger.info(
                "grid_gpu[island]: bright_mask DET-routing: %d unmatched "
                "detection(s) added as DET island members",
                int(bmd.shape[0]))
            (kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar,
             n_det, fit_to_global) = _append_det(
                kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar,
                bmd[:, 0], bmd[:, 1], bmd[:, 2])
        # ---- Round 0 (the only round since card 018 removed the #34
        #      detect_unattributed refit). ----
        ires, shard_stats, island_set, t_island, residuals = \
            _decompose_and_fit(
                kx, ky, kst, klr, ke1, ke2, ksn, kflux, kvar,
                n_det=n_det, want_residual=False, round_idx=0)

    finally:
        for shm in _keepalive:
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Fold the island results back into full-catalog arrays + build the
    # pass-2 (S1' final solve) tasks at the converged parameters.
    # ------------------------------------------------------------------
    # #34: map fit-set rows -> global catalog rows; DET members map to -1
    # and are DROPPED here (they de-railed their neighbours in the joint
    # island fit, and that corrected shape is folded back onto the real
    # sources below; DET rows are not emitted as output rows in this build).
    # When no DET were added (n_det=0), fit_to_global == keep_idx and _vm is
    # all-True -> byte-identical to the pre-#34 fold-back.
    gsrc_all = fit_to_global[ires.source_idx]
    _vm = gsrc_all >= 0
    gsrc = gsrc_all[_vm]
    pos_x_new = xpix.astype(np.float32).copy()
    pos_y_new = ypix.astype(np.float32).copy()
    logre_new = logre_init.astype(np.float32).copy()
    ee1_new = ee1_init.astype(np.float32).copy()
    ee2_new = ee2_init.astype(np.float32).copy()
    sersic_new = sersic_n_init.astype(np.float32).copy()
    pos_x_new[gsrc] = ires.pos_x[_vm]
    pos_y_new[gsrc] = ires.pos_y[_vm]
    logre_new[gsrc] = ires.logre[_vm]
    ee1_new[gsrc] = ires.ee1[_vm]
    ee2_new[gsrc] = ires.ee2[_vm]
    sersic_new[gsrc] = ires.sersic_n[_vm]

    meta = dict(
        island_id=np.full(n_total, -1, dtype=np.int32),
        island_nsrc=np.zeros(n_total, dtype=np.int32),
        island_giant=np.zeros(n_total, dtype=bool),
        island_stalled=np.zeros(n_total, dtype=bool),
        # #35: sticky frozen-proposal-stall disclosure (clip-saturated
        # runaway class). Subset of island_stalled under the default
        # stall_count_mode; never enters fit_is_reliable.
        island_diverged=np.zeros(n_total, dtype=bool),
        island_stall_dchi2=np.full(n_total, np.nan, dtype=np.float32),
        island_railed=np.zeros(n_total, dtype=bool),
        converged=np.zeros(n_total, dtype=bool),
        n_steps=np.zeros(n_total, dtype=np.int32),
        free_n=np.zeros(n_total, dtype=bool),
        fitted_mask=np.zeros(n_total, dtype=bool),
        # LEG D nonlinear (task #20): per-island reduced chi2 (NaN off-island
        # / giants / dof<=0). Feeds the nonlinear *_err_tot columns.
        chi2_nu=np.full(n_total, np.nan, dtype=np.float32),
        # Per-source fitted-Re ceiling (0.75 * stamp tier; NaN off-island
        # and for giants). Feeds the tier-aware flag_shape_bound.
        re_upper_pix=np.full(n_total, np.nan, dtype=np.float32),
        # #28a nonlinear-parameter 1-sigma errors (all-NaN when
        # param_errors_enabled is false or where undefined).
        pos_x_err=np.full(n_total, np.nan, dtype=np.float32),
        pos_y_err=np.full(n_total, np.nan, dtype=np.float32),
        re_err=np.full(n_total, np.nan, dtype=np.float32),
        re_prof_lo=np.full(n_total, np.nan, dtype=np.float32),
        re_prof_hi=np.full(n_total, np.nan, dtype=np.float32),
        ell_err=np.full(n_total, np.nan, dtype=np.float32),
        theta_err=np.full(n_total, np.nan, dtype=np.float32),
        sersic_n_err=np.full(n_total, np.nan, dtype=np.float32),
        # R1 prior_frac disclosure (per prior-carrying fitted param;
        # all-NaN off-island / no-prior / giant). `priors_active` gates
        # the OUTPUT columns so a no-prior run stays byte-identical.
        priors_active=priors_active,
        pos_x_prior_frac=np.full(n_total, np.nan, dtype=np.float32),
        pos_y_prior_frac=np.full(n_total, np.nan, dtype=np.float32),
        re_prior_frac=np.full(n_total, np.nan, dtype=np.float32),
        ell_prior_frac=np.full(n_total, np.nan, dtype=np.float32),
        theta_prior_frac=np.full(n_total, np.nan, dtype=np.float32),
        sersic_n_prior_frac=np.full(n_total, np.nan, dtype=np.float32),
        # #17 per-source model-quality metrics (all-NaN when
        # source_metrics_enabled is false / off-island / giants). The pipeline
        # gates the OUTPUT columns on the config knob, so an off run is
        # byte-identical (columns absent).
        source_metrics_active=source_metrics,
        rchisq=np.full(n_total, np.nan, dtype=np.float32),
        fracflux=np.full(n_total, np.nan, dtype=np.float32),
        fracin=np.full(n_total, np.nan, dtype=np.float32),
        fracmasked=np.full(n_total, np.nan, dtype=np.float32),
        # #35 Option A: per-band data-constrained window fluxes
        # (FLUX_WIN_{band}_fit). Band order = flux_win_bands (the island
        # stage's PSF-carrying band list, a subset of the run's bands).
        # Emission is gated on the same source_metrics knob.
        flux_win=np.full((n_total, n_bands), np.nan, dtype=np.float32),
        flux_win_bands=list(bandnames_with_psf),
        # D2: per-band SHAPE-marginalized flux-error INCREMENT variance
        # (n_total, n_bands). 0 for star rows; NaN off-island / giants /
        # when marginal_flux_err is off. flux_marg_active gates the OUTPUT
        # FLUXERR_MARG* columns so an off run is byte-identical (absent).
        flux_marg_active=marginal_flux_err,
        flux_marg_var=np.full((n_total, n_bands), np.nan, dtype=np.float32),
        wall_island_s=t_island,
        bucket_stats=list(ires.bucket_stats),
        shard_stats={k: v for k, v in shard_stats.items()
                     if k != "per_task"},
    )
    meta["island_id"][gsrc] = ires.island_id[_vm]
    meta["island_nsrc"][gsrc] = ires.island_nsrc[_vm]
    meta["island_giant"][gsrc] = ires.island_giant[_vm]
    meta["island_stalled"][gsrc] = ires.stalled[_vm]
    meta["island_diverged"][gsrc] = ires.diverged[_vm]
    if ires.stall_dchi2 is not None:
        meta["island_stall_dchi2"][gsrc] = ires.stall_dchi2[_vm]
    meta["island_railed"][gsrc] = ires.railed[_vm]
    meta["converged"][gsrc] = ires.converged[_vm]
    meta["n_steps"][gsrc] = ires.n_steps[_vm]
    meta["free_n"][gsrc] = ires.free_n[_vm]
    meta["fitted_mask"][gsrc] = True
    if ires.chi2_nu is not None:
        meta["chi2_nu"][gsrc] = ires.chi2_nu[_vm]
    meta["re_upper_pix"][gsrc] = ires.re_upper_pix[_vm]
    meta["pos_x_err"][gsrc] = ires.pos_x_err[_vm]
    meta["pos_y_err"][gsrc] = ires.pos_y_err[_vm]
    meta["re_err"][gsrc] = ires.re_err[_vm]
    meta["re_prof_lo"][gsrc] = ires.re_prof_lo[_vm]
    meta["re_prof_hi"][gsrc] = ires.re_prof_hi[_vm]
    meta["ell_err"][gsrc] = ires.ell_err[_vm]
    meta["theta_err"][gsrc] = ires.theta_err[_vm]
    meta["sersic_n_err"][gsrc] = ires.sersic_n_err[_vm]
    meta["pos_x_prior_frac"][gsrc] = ires.pos_x_prior_frac[_vm]
    meta["pos_y_prior_frac"][gsrc] = ires.pos_y_prior_frac[_vm]
    meta["re_prior_frac"][gsrc] = ires.re_prior_frac[_vm]
    meta["ell_prior_frac"][gsrc] = ires.ell_prior_frac[_vm]
    meta["theta_prior_frac"][gsrc] = ires.theta_prior_frac[_vm]
    meta["sersic_n_prior_frac"][gsrc] = ires.sersic_n_prior_frac[_vm]
    if getattr(ires, "rchisq", None) is not None:
        meta["rchisq"][gsrc] = ires.rchisq[_vm]
        meta["fracflux"][gsrc] = ires.fracflux[_vm]
        meta["fracin"][gsrc] = ires.fracin[_vm]
        meta["fracmasked"][gsrc] = ires.fracmasked[_vm]
    if getattr(ires, "flux_win", None) is not None:
        meta["flux_win"][gsrc] = ires.flux_win[_vm]
    if getattr(ires, "flux_marg_var", None) is not None:
        meta["flux_marg_var"][gsrc] = ires.flux_marg_var[_vm]

    # Pass-2 tasks: same grids/data/stamps, S1-only again, at the
    # converged nonlinear params (big-galaxy wings in this native-stamp
    # solve are covered by the production wing-peeling mechanism, which
    # re-selects its big sources from the FITTED shapes).
    # Exp C v2: protection-demoted phantoms are excluded from the pass-2
    # final solve entirely -- a demoted "detection" is a cataloged
    # galaxy's own light, and a PSF column left on it would absorb part
    # of that galaxy's REPORTED flux.
    _br_flag = bool(island_cfg.get("blank_response_correction", False))
    _dropg = (np.unique(np.asarray(demoted_glob, dtype=np.int64))
              if demoted_glob else None)
    if _dropg is not None and _dropg.size:
        logger.info("grid_gpu[island]: Exp C v2: excluding %d demoted "
                    "phantom column(s) from the pass-2 final solve",
                    int(_dropg.size))
    pass2 = []
    for t in tasks:
        fi = t.source_indices
        if _dropg is not None and _dropg.size and np.isin(fi, _dropg).any():
            _m = ~np.isin(fi, _dropg)
            pass2.append(dataclasses.replace(
                t,
                source_indices=fi[_m],
                source_type=t.source_type[_m],
                pos_err_pix=t.pos_err_pix[_m],
                owned_source_mask=(t.owned_source_mask[_m]
                                   if t.owned_source_mask is not None
                                   else None),
                pos_x_init=pos_x_new[fi][_m],
                pos_y_init=pos_y_new[fi][_m],
                logre_init=logre_new[fi][_m],
                ee1_init=ee1_new[fi][_m],
                ee2_init=ee2_new[fi][_m],
                sersic_n_init=sersic_new[fi][_m],
                blank_response_correction=_br_flag,
            ))
            continue
        pass2.append(dataclasses.replace(
            t,
            pos_x_init=pos_x_new[fi],
            pos_y_init=pos_y_new[fi],
            logre_init=logre_new[fi],
            ee1_init=ee1_new[fi],
            ee2_init=ee2_new[fi],
            sersic_n_init=sersic_new[fi],
            blank_response_correction=_br_flag,
        ))
    logger.info("grid_gpu[island]: pass-2 (S1' final solve) tasks built "
                "at the converged parameters")

    # ---- blend_resolve R3: sentinel star columns in the pass-2 solve ----
    # The reported FLUX comes from the pass-2 S1' flux re-solve on the RAW
    # grid data -- a resolved star absent from that solve gets RE-ABSORBED
    # into its host's flux, undoing the island-stage recovery. Pass-2 is an
    # all-frozen-positions flux-only solve, so the exact fix is one extra
    # STAR column per ACCEPTED resolved star, carried with a NEGATIVE
    # sentinel source index: the pipeline writeback strips sentinel rows
    # (capturing their solved fluxes for blend_resolved.csv) and they are
    # never emitted as catalog rows. Ledger (all candidates, accepted or
    # not) -> {work_dir}/blend_resolved.csv; accepted hosts additionally
    # tagged via meta["blend_resolved"].
    meta["blend_resolved"] = np.zeros(n_total, dtype=bool)
    meta["blend_unresolved"] = np.zeros(n_total, dtype=bool)
    if br_resolved:
        _accs = [r for r in br_resolved if r.get("accepted")]
        for _q, _r3 in enumerate(_accs):
            _r3["sent_id"] = -(2 + _q)
            meta["blend_resolved"][int(_r3["host_global"])] = True
        # gate-passing candidate + repair REJECTED = blend_unresolved
        # (positive uncorrectable-contamination evidence; enters
        # fit_is_reliable per merge.blend_unresolved_reliability,
        # default hard-fail -- user decision 2026-07-14).
        for _r3 in br_resolved:
            if not _r3.get("accepted"):
                meta["blend_unresolved"][int(_r3["host_global"])] = True
        _own_of = {}
        for _ti, t in enumerate(pass2):
            _om = t.owned_source_mask
            for _p, _g in enumerate(np.asarray(t.source_indices)):
                if _om is None or bool(_om[_p]):
                    _own_of[int(_g)] = _ti
        _by_task: dict[int, list] = {}
        for _r3 in _accs:
            _ti = _own_of.get(int(_r3["host_global"]))
            if _ti is None:
                logger.warning(
                    "blend_resolve: no owning pass-2 task for host "
                    "global %d; star column skipped (its pass-2 flux "
                    "will re-absorb)", int(_r3["host_global"]))
                continue
            _by_task.setdefault(_ti, []).append(_r3)
        for _ti, _r3s in _by_task.items():
            t = pass2[_ti]
            _ns = len(_r3s)
            _sx = np.array([r["cand_x"] for r in _r3s], np.float64)
            _sy = np.array([r["cand_y"] for r in _r3s], np.float64)
            _sid = np.array([r["sent_id"] for r in _r3s], np.int64)
            pass2[_ti] = dataclasses.replace(
                t,
                source_indices=np.concatenate(
                    [np.asarray(t.source_indices, np.int64), _sid]),
                source_type=np.concatenate(
                    [np.asarray(t.source_type, np.int32),
                     np.zeros(_ns, np.int32)]),
                pos_err_pix=np.concatenate(
                    [np.asarray(t.pos_err_pix, np.float64),
                     np.full(_ns, 1.0)]),
                owned_source_mask=(np.concatenate(
                    [np.asarray(t.owned_source_mask, bool),
                     np.ones(_ns, bool)])
                    if t.owned_source_mask is not None else None),
                pos_x_init=np.concatenate(
                    [np.asarray(t.pos_x_init, np.float64), _sx]),
                pos_y_init=np.concatenate(
                    [np.asarray(t.pos_y_init, np.float64), _sy]),
                logre_init=np.concatenate(
                    [np.asarray(t.logre_init, np.float32),
                     np.zeros(_ns, np.float32)]),
                ee1_init=np.concatenate(
                    [np.asarray(t.ee1_init, np.float32),
                     np.ones(_ns, np.float32)]),
                ee2_init=np.concatenate(
                    [np.asarray(t.ee2_init, np.float32),
                     np.zeros(_ns, np.float32)]),
                sersic_n_init=(np.concatenate(
                    [np.asarray(t.sersic_n_init, np.float32),
                     np.ones(_ns, np.float32)])
                    if t.sersic_n_init is not None else None),
            )
        if work_dir is not None:
            try:
                import pandas as _pd4
                _led = [{k: v for k, v in r.items()
                         if not k.startswith("_")} for r in br_resolved]
                _pd4.DataFrame(_led).to_csv(
                    os.path.join(str(work_dir), "blend_resolved.csv"),
                    index=False)
            except Exception as _e4:
                logger.warning("blend_resolved.csv write failed: %s", _e4)
        logger.info(
            "grid_gpu[island]: blend_resolve: %d/%d candidate(s) accepted "
            "-> %d pass-2 sentinel star column(s) across %d task(s)",
            len(_accs), len(br_resolved),
            sum(len(v) for v in _by_task.values()), len(_by_task))
    return pass2, meta


__all__ = ["run_island_prestage", "ISLAND_BBOX_CLASSES",
           "ISLAND_NSRC_CLASSES"]
