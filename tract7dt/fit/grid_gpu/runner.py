"""Per-grid worker: runs the 3-stage LM fit on ONE image quadrant on
ONE GPU.

Designed to be invoked via :class:`concurrent.futures.ProcessPoolExecutor`
with one worker per GPU. The worker initializer pins
``CUDA_VISIBLE_DEVICES`` so JAX sees only this worker's device.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .shm_hygiene import start_parent_watchdog


logger = logging.getLogger("tract7dt.fit.grid_gpu.runner")


# Module-level container for state that should persist across worker
# tasks (when using a long-lived ProcessPool worker). The persistent
# JIT cache lives at the JAX library level; we just keep references
# to the per-band PsfPolynomial objects and the catalog cross-walk
# here to avoid recomputing them per grid.
_WORKER_STATE: dict[str, Any] = {}


def enable_persistent_jit_cache(jit_cache_dir: Any) -> "str | None":
    """Enable the persistent XLA compilation cache in THIS process.

    Resolves the ``compute.gpu.jit_cache_dir`` config value (``"auto"`` ->
    ``~/.cache/tract7dt/jax_jit_cache_grid``; ``None``/``"none"`` -> disabled)
    and points JAX at it, with the min-compile-time threshold at 0 so the
    ~78 SMALL eager render-path programs (<1s compile each) persist too --
    JAX's default threshold skips them, so every grid in every run paid
    ~3.0s recompiling them (measured: first-call S1 delta 3.05s -> 0.44s
    with the threshold at 0). Byte-identical executables; cache grows a few
    hundred MB per tile.

    ONE implementation for BOTH callers -- the grid workers (:func:`_worker_init`)
    and the parent-side grid-peak probe (``pipeline._measured_grid_peak_mib``,
    G1) -- so the two cannot drift apart (the T7 lesson: two call sites, one
    parameter home). Returns the resolved cache dir, or None when disabled.
    Raises on JAX API failure; callers log-and-continue."""
    if jit_cache_dir is None or str(jit_cache_dir) == "none":
        return None
    cache_dir = (
        os.path.expanduser("~/.cache/tract7dt/jax_jit_cache_grid")
        if str(jit_cache_dir) == "auto" else str(jit_cache_dir)
    )
    from jax.experimental import compilation_cache as cc
    if hasattr(cc, "set_cache_dir"):
        cc.set_cache_dir(cache_dir)
    else:
        cc.compilation_cache.set_cache_dir(cache_dir)
    import jax
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    return cache_dir


def _worker_ping() -> int:
    """Lightweight task used to start a pinned GPU worker before large payloads.

    The ProcessPool initializer has already selected the GPU and warmed JAX by
    the time this runs. Submitting this tiny task lets the parent bring all GPU
    workers up in parallel before it serializes multi-GB grid inputs.
    """
    return int(_WORKER_STATE.get("gpu_device_id", -1))


def _worker_init(
    gpu_device_id: int,
    mem_fraction: float,
    jit_cache_dir: str | None,
    log_file_path: str | None = None,
    alloc_mode: str = "platform",
) -> None:
    """Per-worker initializer: pin GPU device + warm JAX context.

    Sets ``argv[0]`` to a recognizable string ("tract7dt-grid-worker")
    so ``pkill -f tract7dt`` catches both the parent and these worker
    subprocesses cleanly. Without this, multiprocessing.spawn workers
    show up as ``python -c 'from multiprocessing.spawn import ...'``,
    which doesn't match the ``tract7dt`` pattern and leaves zombies
    holding GPU memory after the parent is killed.
    

    Memory allocator: we use ``XLA_PYTHON_CLIENT_ALLOCATOR=platform``,
    which makes JAX call ``cudaMalloc/cudaFree`` directly per allocation
    instead of using the BFC arena allocator. This is critical for our
    workload because:

    - The 3-stage LM solver alternates between many small per-source
      ops and a few large per-band linear solves; the BFC arena tends
      to over-reserve and then OOM on transient peaks.
    - Each per-grid worker is the only consumer of its dedicated GPU
      (one process per GPU), so there's no fragmentation pressure
      across processes.
    - Slightly higher per-call malloc cost (~microseconds) is dwarfed
      by the per-grid solve time (~minutes).

    With the platform allocator, ``XLA_PYTHON_CLIENT_PREALLOCATE`` and
    ``XLA_PYTHON_CLIENT_MEM_FRACTION`` are effectively ignored; we
    retain them for potential fallback to the BFC allocator.
    """
    # FIRST: exit when the parent dies. A spawn worker holds both ends of
    # its call-queue pipe, so after a parent SIGKILL (kernel OOM killer) it
    # would otherwise block in call_queue.get() forever -- pinning GPU/host
    # memory, its semaphores, and the resource_tracker pipe (which blocks
    # the tracker's emergency unlink of the run's shared memory).
    start_parent_watchdog()

    # Make this subprocess findable by `pkill -f tract7dt-grid-worker`
    # (multiprocessing.spawn workers default to `python -c "..."` which
    # is hard to target). Best-effort; fails silently on systems where
    # setproctitle isn't installed.
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(f"tract7dt-grid-worker-gpu{gpu_device_id}")
    except ImportError:
        pass

    # Configure logging in this spawned subprocess. Spawn workers do
    # NOT inherit the parent's logging config; without a handler,
    # INFO-level messages from this worker would be silently dropped
    # (Python's "lastResort" handler only kicks in for WARNING+).
    # We attach a stderr handler so worker logs flow back to the terminal,
    # and optionally a file handler so per-stage worker timings are present
    # in the same run.log as the parent process.
    root = logging.getLogger()
    if not any(getattr(h, "_tract7dt_worker_handler", False) for h in root.handlers):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        ))
        h._tract7dt_worker_handler = True  # type: ignore[attr-defined]
        root.addHandler(h)
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        want = str(log_path.resolve())
        has_file = False
        for h in root.handlers:
            if getattr(h, "_tract7dt_worker_file_handler", False):
                if str(Path(getattr(h, "baseFilename", "")).resolve()) == want:
                    has_file = True
                    break
        if not has_file:
            fh = logging.FileHandler(log_path, mode="a")
            fh.setFormatter(logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            ))
            fh._tract7dt_worker_file_handler = True  # type: ignore[attr-defined]
            root.addHandler(fh)
    root.setLevel(logging.INFO)

    # FORCE-set (not setdefault) so any env vars inherited from the parent
    # process are overridden in this worker. These MUST be set before
    # `import jax` because JAX reads them at module-import time and
    # caches the allocator decision.
    #
    # 2026-08-15 fix: `gpu_device_id` is the PARENT's jax-visible device
    # index. If the parent itself ran under a CUDA_VISIBLE_DEVICES
    # restriction, writing the index NUMBER here silently re-pointed the
    # worker at a different PHYSICAL card (measured: a parent restricted to
    # GPUs 1,2,3 spawned its id-0 worker onto physical GPU 0, which an
    # external job had filled -- the run died on a 97 MiB allocation and
    # the fast-failing worker ate 6 of 8 queue tasks). Map the index
    # through the inherited restriction; with no restriction this is the
    # identity mapping (str(id)), byte-identical to the old behavior.
    _parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if _parent_cvd:
        _vis = [x.strip() for x in _parent_cvd.split(",") if x.strip()]
        try:
            _phys = _vis[int(gpu_device_id)]
        except (IndexError, ValueError):
            _phys = str(gpu_device_id)
    else:
        _phys = str(gpu_device_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = _phys
    os.environ["JAX_PLATFORMS"] = "cuda"
    os.environ["JAX_ENABLE_X64"] = "1"
    # PRECISION (2026-08-30, development ledger SS23-25/32 + SS24): XLA silently
    # demotes every float32 dot/einsum/matmul to TF32 -- a 10-bit mantissa
    # instead of 23 -- unless told otherwise. That reaches the REPORTED numbers:
    # the compiled island kernels (jit_final_fluxes, jit_final_errors,
    # jit_source_metrics, jit_one_step) carry `mma.sync...f32.tf32.tf32.f32`
    # PTX in the shipped JIT cache. Measured on the sparse anchor T08676 with
    # `highest`: tf32 MMA instructions 1,884 -> 0, wall 639/739 s against a
    # 715/608 s baseline pair (no cost), reported flux moved by 0.040 mmag
    # (median) / 0.51 mmag (p99), and -- the reason this is not merely a
    # correctness nicety -- the cold-run reproducibility went from 48.8%
    # bit-identical / 13.65 mmag max / 8 reliability flips to 90.6% / 0.0004
    # mmag / 0 flips. The lineage lottery this arc spent two months wrapping in
    # envelopes was, on that anchor, autotuned Triton TF32 gemms.
    # This is NOT a tuned constant: on this hardware (RTX 3090 / GA102) dense
    # TF32 tensor throughput EQUALS FP32 CUDA throughput, so the demotion buys
    # exactly nothing and costs mantissa. `setdefault`, not a hard set, so an
    # operator can A/B it (that is how it was measured) and so a datacenter GPU
    # where TF32 really is faster can opt back in deliberately.
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    if str(alloc_mode).lower() == "bfc":
        # Capped BFC arena: HARD-bounds this worker's GPU memory to
        # `mem_fraction` of the card so multiple workers packed on one GPU
        # cannot collectively exceed it (the platform allocator below is
        # on-demand and uncapped, which is why naive packing collided/OOM'd).
        # A grid whose peak exceeds the cap OOMs cleanly within its arena and
        # is retried on an exclusive GPU by the dispatcher -> never corrupts
        # output. Used only when workers_per_gpu > 1.
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        if mem_fraction > 0:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{float(mem_fraction):.4f}"
    else:
        # On-demand platform allocator (no cap): the default for a SOLO worker
        # that owns the whole GPU (no co-location risk). The platform allocator
        # does per-allocation cudaMalloc and IGNORES XLA_PYTHON_CLIENT_MEM_FRACTION
        # (that only sizes the BFC arena), so we do NOT set it -- the worker
        # allocates on demand up to the full card. Clear any value inherited from
        # the parent so the log/env honestly reflects "uncapped".
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ.pop("XLA_PYTHON_CLIENT_MEM_FRACTION", None)
    logger.info(
        "worker_init: GPU=%d, CUDA_VISIBLE_DEVICES=%s, alloc_mode=%s, "
        "ALLOCATOR=%s, PREALLOCATE=%s, MEM_FRACTION=%s, MPS=%s",
        gpu_device_id,
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        alloc_mode,
        os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "<unset>"),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "<unset>"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "<unset>"),
        os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "<none>"),
    )
    # Pre-import + warm context (so the first patch fit doesn't pay
    # the CUDA-context-creation tax, ~3-15 s).
    try:
        import jax
        import jax.numpy as jnp
        # Touch the device: forces context creation. Verify allocator.
        _warm = jax.jit(lambda x: x * 2.0)(jnp.arange(8, dtype=jnp.float32))
        _warm.block_until_ready()
        try:
            backend = jax.default_backend()
            n_devices = jax.device_count()
            logger.info(
                "worker_init: JAX backend=%s, n_devices=%d (after CUDA_VISIBLE_DEVICES)",
                backend, n_devices,
            )
        except Exception:
            pass
        # Configure persistent JIT cache (process-wide).
        try:
            cache_dir = enable_persistent_jit_cache(jit_cache_dir)
            if cache_dir is not None:
                logger.info("JIT persistent cache enabled at: %s", cache_dir)
        except Exception as exc:
            logger.warning("Failed to set JIT cache: %s", exc)
        # Pre-import grid_gpu modules so the first task doesn't pay
        # the import cost.
        import tract7dt  # noqa: F401
        import tract7dt.fit.grid_gpu.solver_grid  # noqa: F401
    except Exception as exc:
        logger.warning("Worker init warmup failed (non-fatal): %s", exc)

    # ---- verify the two numerical settings actually took effect -----------
    # DELIBERATELY OUTSIDE the warm-up try/except above: that block downgrades
    # every failure to a warning ("non-fatal"), which is right for a context
    # warm-up and exactly wrong for this. Both settings are env vars JAX reads
    # at import time, so "we set it a few lines up" is not evidence -- an
    # inherited value, an import that moved earlier in a refactor, or a
    # renamed variable all fail SILENTLY and the run just gets less precise.
    # The engine's only other runtime x64 guard sits inside the OPT-IN
    # `island.convergence: "dlnp"` branch (island_solver.py), i.e. it never
    # fires on the shipped default, while ~26 `astype(float64)` sites in that
    # file would quietly become float32. This is the one place every grid and
    # island shard worker passes through with JAX already imported.
    try:
        import jax  # noqa: PLC0415  (already imported above; cached)
        _x64 = bool(jax.config.jax_enable_x64)
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            f"worker_init: cannot import JAX to verify numerics: {exc}"
        ) from exc
    if not _x64:
        raise RuntimeError(
            "worker_init: JAX_ENABLE_X64 did not take effect in this worker "
            "-- every float64 accumulator and the LU solve would silently "
            "run in float32. Refusing to produce photometry.")
    _prec = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "<unset>")
    logger.info("worker_init: x64=%s, matmul_precision=%s", _x64, _prec)
    if _prec != "highest":
        # Not a hard failure: this one is a `setdefault`, so an operator may
        # trade it away on hardware where TF32 is genuinely faster (it is NOT
        # on GA102 -- dense TF32 throughput equals FP32 there). But it must be
        # VISIBLE, because the damage is deterministic and therefore invisible
        # to every run-to-run gate this arc has ever run.
        logger.warning(
            "worker_init: JAX_DEFAULT_MATMUL_PRECISION=%s -- XLA may demote "
            "float32 dots to TF32 (10-bit mantissa). Measured on the sparse "
            "anchor: reported-flux p99 0.51 mmag and a 36,000x wider "
            "run-to-run envelope (development ledger SS24).", _prec)

    _WORKER_STATE["gpu_device_id"] = int(gpu_device_id)
    _WORKER_STATE["initialized"] = True


@dataclass
class SharedArraySlice:
    """Descriptor for a 2-D slice backed by multiprocessing shared memory."""
    shm_name: str
    shape: tuple[int, int]
    dtype: str
    y0: int
    y1: int
    x0: int
    x1: int


def _maybe_pad_psf(psf_obj: Any, target_shape: tuple) -> Any:
    """Zero-pad a PsfPolynomial's stamp to a larger (H, W) tier.

    Phase-B stamp promotion: everything downstream (galaxy render box,
    scatter windows, neighbour predicate, Re bounds) keys off
    ``psf_jax.H_psf/W_psf``, so enlarging the model stamp for grids that
    contain large galaxies reduces to padding the PSF stamp with zeros
    (the PSF itself stays compact and unit-sum). No-op when the target
    equals the native size or for PSF objects without a ``coefs`` array
    (e.g. per-cell lookups, which are not used with stamp promotion).
    """
    th, tw = int(target_shape[0]), int(target_shape[1])
    H = int(getattr(psf_obj, "H_psf", 0))
    W = int(getattr(psf_obj, "W_psf", 0))
    coefs = getattr(psf_obj, "coefs", None)
    if coefs is None or (th <= H and tw <= W):
        return psf_obj
    import dataclasses
    pad_y0 = (th - H) // 2
    pad_x0 = (tw - W) // 2
    new_coefs = np.zeros((th, tw) + coefs.shape[2:], dtype=coefs.dtype)
    new_coefs[pad_y0:pad_y0 + H, pad_x0:pad_x0 + W] = coefs
    return dataclasses.replace(
        psf_obj, coefs=new_coefs, H_psf=th, W_psf=tw)


@dataclass
class PerGridTask:
    """Input to a per-grid worker: everything needed to fit ONE quadrant.

    The worker assembles :class:`GridMultiBandInputs` from this and
    calls :func:`solve_grid_3stage`. The PSF polynomials are
    serializable via numpy + simple float fields, so they pickle
    cleanly across processes.
    """
    grid_id: int
    grid_row: int
    grid_col: int
    # Per-band image arrays (already sliced to this grid's padded bbox).
    bandnames: list[str]
    data_per_band: list[np.ndarray]      # (band_h, band_w) per band
    invvar_per_band: list[np.ndarray]
    pix_good_per_band: list[np.ndarray]
    # Per-band PSF model. Polymorphic: each entry is either a
    # `PsfPolynomial` (PSFEx-style polynomial fit) or a
    # `PsfPerCellLookup` (nearest-cell ePSF lookup). The worker calls
    # `.to_jax()` to get the JAX-friendly version. Both implement the
    # `evaluate_psf_at(psf_jax, x, y)` interface (see psf_polynomial.py).
    psfs_per_band: list[Any]
    psf_shape: tuple[int, int]           # (H_psf, W_psf), all bands share
    # Grid offset in full-image pixel coords.
    grid_x0: int
    grid_y0: int
    # Source subset (rows owned by this grid in the full input catalog).
    source_indices: np.ndarray            # int64; positions into full catalog
    pos_x_init: np.ndarray                # (n_sources,) full-image pixel coords
    pos_y_init: np.ndarray
    pos_err_pix: np.ndarray               # (n_sources,) Gaussian prior sigma
    source_type: np.ndarray               # int32; 0=star, 1=exp, 2=dev, 3=sersic
    logre_init: np.ndarray
    ee1_init: np.ndarray
    ee2_init: np.ndarray
    # Solver knobs (config-driven).
    pos_max_shift_pix: float
    bg_x_origin: float
    bg_x_scale: float
    bg_y_origin: float
    bg_y_scale: float
    bg_poly_order: int = 2
    single_stage: bool = False
    n_opt_iters: int = 5
    pos_step_max_pix: float = 1.0
    dlnp_stop_pix: float = 0.01
    # Dataclass default only; every GridTask is constructed with an explicit
    # position_solver plumbed from the (load_config-merged) cfg. DEFAULT_CONFIG
    # is the single source of truth for the shipped default (currently "island").
    position_solver: str = "per_source"
    joint_cg_max_iters: int = 20
    joint_cg_tol: float = 1.0e-6
    joint_use_preconditioner: bool = True
    pos_stop_unconverged_frac: float = 0.0
    chi2_ftol: float = 0.1
    # Extended-source model: per-source Sersic index (DEV/SERSIC rendering)
    # + renderer choice ("direct" | "fft"), resolved by the pipeline.
    sersic_n_init: np.ndarray | None = None
    render_backend: str = "direct"
    # Stage-2 free-n (varpro): fit the Sersic index of TYPE=SERSIC rows
    # whose S1 flux S/N clears n_fit_min_snr; the rest stay frozen at
    # the seed (compute.gpu.galaxy.fit_sersic_n / n_fit_min_snr).
    fit_sersic_n: bool = True
    n_fit_min_snr: float = 30.0
    # Wing peeling for large galaxies (compute.gpu.galaxy.wing_peeling /
    # wing_peel_tier / wing_peel_max_sources): subtract per-band wing
    # images (tier render minus native render) of the few big sources from
    # the data before every solve; solver machinery stays at the native
    # stamp (see solver_grid wing-peeling block).
    wing_peeling: bool = False
    wing_peel_tier: int = 221
    wing_peel_max_sources: int = 64
    # Blank-response (empty-aperture) median flux-bias correction
    # (island.blank_response_correction): pass-2 only (see solver_grid).
    blank_response_correction: bool = False
    # Post-solve confusion error measurement (task #27; error_model.py):
    # compute sigma_conf per source per band from the residual map at the
    # converged state. The pipeline enables this only on the FINAL solve
    # pass (island pass-1 seed solves keep it off -- their results are
    # discarded). Config: compute.gpu.error_model.confusion_enabled.
    compute_confusion: bool = False
    # Post-solve LEG C crowded-robust residual F_corr (task #20): per-band
    # correlated-noise factor from this grid's residual (error_model.py),
    # combined across grids by the parent. Enabled on the FINAL solve pass
    # when compute.gpu.error_model.correlated_noise_enabled.
    compute_fcorr_resid: bool = False
    # How many grid workers run CONCURRENTLY on this machine (len(devices) x
    # workers_per_gpu). Set by the parent once packing is decided; divides the
    # core/memory budget of the host-side band pool in error_model so sibling
    # workers cannot collectively oversubscribe. 1 = sole occupant.
    n_grid_slots: int = 1
    shared_data_per_band: list[SharedArraySlice] = field(default_factory=list)
    shared_invvar_per_band: list[SharedArraySlice] = field(default_factory=list)
    shared_pix_good_per_band: list[SharedArraySlice] = field(default_factory=list)
    owned_source_mask: np.ndarray | None = None  # True for rows written by this grid


@dataclass
class PerGridResult:
    """Output from a per-grid worker."""
    grid_id: int
    n_sources: int
    pos_x: np.ndarray                    # (n_sources,)
    pos_y: np.ndarray
    logre: np.ndarray
    ee1: np.ndarray
    ee2: np.ndarray
    fluxes: np.ndarray                   # (n_sources, n_bands)
    bg_coefs: np.ndarray                 # (n_bands, M)
    flux_var: np.ndarray
    bg_var: np.ndarray
    pos_var: np.ndarray
    source_indices: np.ndarray           # passed through for catalog rejoin
    s1_converged: bool
    s2_converged: bool
    s3_converged: bool
    n_outer_iters: int
    sec_total: float
    error: str | None = None
    # Per-source: position step still above tolerance when S3 stopped
    # (quantile stop); see solver_grid.GridMultiBandResult.
    position_stranded: np.ndarray | None = None
    # Stage-2 free-n: final Sersic index (fitted for free-n SERSIC rows,
    # seed otherwise) + freeze flag (TYPE==SERSIC and n NOT freed) +
    # 1-sigma n uncertainty (free rows only, NaN elsewhere; diagonal /
    # flux-marginalized -- see solver_grid._varpro_sersic_n_err).
    sersic_n_fit: np.ndarray | None = None
    sersic_n_frozen: np.ndarray | None = None
    sersic_n_err: np.ndarray | None = None
    # Per-source: True for a DEV/SERSIC row whose grid ran a shape-freezing
    # solver (per_source/joint_lsqr) -- its Re/ELL/THETA are the catalog
    # seed, not a fit (informational flag; see solver_grid.shape_frozen).
    shape_frozen: np.ndarray | None = None
    # Measured per-source confusion noise (n_sources, n_bands) float32,
    # NaN where unavailable; None when the measurement was disabled or
    # failed (task #27; see error_model.py). Feeds the FLUXERR_CONF_* /
    # FLUXERR_TOT_* catalog columns.
    fluxerr_conf: np.ndarray | None = None
    # LEG C crowded-robust residual F_corr (task #20): per-band list (length
    # n_bands, ordered like the fit bands) of (F_corr, n_clean, fallback) or
    # None. PER-GRID (not per-source); the parent combines across grids.
    fcorr_resid: list | None = None
    # (n_bands,) blank-response b_hat subtracted from this grid's STAR
    # fluxes, ordered like the fit bands; None when the correction did not
    # run. PER-GRID; the parent writes blank_response_bhat.csv (R09).
    blank_response_bhat: np.ndarray | None = None


def run_one_grid(task: PerGridTask) -> PerGridResult:
    """Worker entry: fit one image grid."""
    t0 = time.time()
    attached_shm: list[Any] = []
    try:
        from .forced_phot_sparse import GridBandInputs
        from .bg_polynomial import make_bg_basis_jax
        from .solver_grid import GridMultiBandInputs, solve_grid_3stage
        import jax
        import jax.numpy as jnp

        def _arrays_from_shared(specs: list[SharedArraySlice]) -> list[np.ndarray]:
            from multiprocessing import shared_memory
            arrays: list[np.ndarray] = []
            for spec in specs:
                shm = shared_memory.SharedMemory(name=spec.shm_name)
                attached_shm.append(shm)
                full = np.ndarray(
                    spec.shape, dtype=np.dtype(spec.dtype), buffer=shm.buf,
                )
                arrays.append(full[spec.y0:spec.y1, spec.x0:spec.x1])
            return arrays

        # Reconstruct per-band JAX PSFs by calling each PsfPolynomial /
        # PsfPerCellLookup's `.to_jax()` method. When the grid is promoted
        # to a larger stamp tier (Phase B), the LINEAR flux solves use the
        # zero-padded stamps while the position/shape steps keep the native
        # ones (see GridMultiBandInputs.psf_jaxes_native).
        psf_jaxes = [
            _maybe_pad_psf(psf_obj, task.psf_shape).to_jax()
            for psf_obj in task.psfs_per_band
        ]
        native_hw = (task.psfs_per_band[0].H_psf, task.psfs_per_band[0].W_psf)
        if tuple(task.psf_shape) != tuple(native_hw):
            psf_jaxes_native = [p.to_jax() for p in task.psfs_per_band]
        else:
            psf_jaxes_native = None

        if task.shared_data_per_band:
            data_per_band = _arrays_from_shared(task.shared_data_per_band)
            invvar_per_band = _arrays_from_shared(task.shared_invvar_per_band)
            pix_good_per_band = _arrays_from_shared(task.shared_pix_good_per_band)
        else:
            data_per_band = task.data_per_band
            invvar_per_band = task.invvar_per_band
            pix_good_per_band = task.pix_good_per_band

        # Build per-band GridBandInputs.
        bands_inputs: list[GridBandInputs] = []
        for bi, (bn, data, invvar, pixg) in enumerate(zip(
            task.bandnames, data_per_band, invvar_per_band, pix_good_per_band,
        )):
            bands_inputs.append(GridBandInputs(
                data=jnp.asarray(data, dtype=jnp.float32),
                invvar=jnp.asarray(invvar, dtype=jnp.float32),
                pix_good=jnp.asarray(pixg, dtype=bool),
                pos_x=jnp.asarray(task.pos_x_init, dtype=jnp.float32),
                pos_y=jnp.asarray(task.pos_y_init, dtype=jnp.float32),
                psf_jax=psf_jaxes[bi],
                band_h=int(data.shape[0]), band_w=int(data.shape[1]),
                grid_x0=int(task.grid_x0), grid_y0=int(task.grid_y0),
                bg_x_origin=float(task.bg_x_origin), bg_x_scale=float(task.bg_x_scale),
                bg_y_origin=float(task.bg_y_origin), bg_y_scale=float(task.bg_y_scale),
                source_type=jnp.asarray(task.source_type, dtype=jnp.int32),
                logre=jnp.asarray(task.logre_init, dtype=jnp.float32),
                ee1=jnp.asarray(task.ee1_init, dtype=jnp.float32),
                ee2=jnp.asarray(task.ee2_init, dtype=jnp.float32),
                sersic_n=jnp.asarray(
                    task.sersic_n_init
                    if task.sersic_n_init is not None
                    else np.ones_like(task.logre_init),
                    dtype=jnp.float32,
                ),
                render_backend=str(task.render_backend),
            ))
        if bands_inputs:
            t_basis = time.time()
            inp0 = bands_inputs[0]
            bg_basis_flat = make_bg_basis_jax(
                inp0.band_h, inp0.band_w, inp0.grid_x0, inp0.grid_y0,
                inp0.bg_x_origin, inp0.bg_x_scale,
                inp0.bg_y_origin, inp0.bg_y_scale,
                order=int(task.bg_poly_order),
            )
            jax.block_until_ready(bg_basis_flat)
            for inp_b in bands_inputs:
                inp_b.bg_basis_flat = bg_basis_flat
            logger.info(
                "grid %d: cached bg basis shape=%s in %.2fs",
                task.grid_id, tuple(bg_basis_flat.shape), time.time() - t_basis,
            )

        # Bounds: pos_init +/- pos_max_shift_pix.
        pos_lower_x = task.pos_x_init - task.pos_max_shift_pix
        pos_upper_x = task.pos_x_init + task.pos_max_shift_pix
        pos_lower_y = task.pos_y_init - task.pos_max_shift_pix
        pos_upper_y = task.pos_y_init + task.pos_max_shift_pix

        multi = GridMultiBandInputs(
            bands=bands_inputs,
            bandnames=list(task.bandnames),
            pos_init_x=task.pos_x_init.astype(np.float32),
            pos_init_y=task.pos_y_init.astype(np.float32),
            pos_err=task.pos_err_pix.astype(np.float32),
            pos_lower_x=pos_lower_x.astype(np.float32),
            pos_upper_x=pos_upper_x.astype(np.float32),
            pos_lower_y=pos_lower_y.astype(np.float32),
            pos_upper_y=pos_upper_y.astype(np.float32),
            source_type=task.source_type.astype(np.int32),
            logre_init=task.logre_init.astype(np.float32),
            ee1_init=task.ee1_init.astype(np.float32),
            ee2_init=task.ee2_init.astype(np.float32),
            logre_lower=np.full(task.logre_init.shape, np.log(0.1), dtype=np.float32),
            # Re upper bound tied to what the stamp can constrain: beyond
            # ~0.75x the stamp width the profile is flat across the render
            # box and degenerate with the background (observed runaway:
            # Re -> 100+ px on invisible LSB sources). A larger stamp tier
            # is the proper lift for genuinely larger galaxies.
            logre_upper=np.full(
                task.logre_init.shape,
                np.log(0.75 * float(task.psf_shape[1])),
                dtype=np.float32,
            ),
            ee1_lower=np.full(task.ee1_init.shape, -100.0, dtype=np.float32),
            ee1_upper=np.full(task.ee1_init.shape, 100.0, dtype=np.float32),
            ee2_lower=np.full(task.ee2_init.shape, -100.0, dtype=np.float32),
            ee2_upper=np.full(task.ee2_init.shape, 100.0, dtype=np.float32),
            sersic_n=(
                task.sersic_n_init.astype(np.float32)
                if task.sersic_n_init is not None
                else np.ones_like(task.logre_init, dtype=np.float32)
            ),
            render_backend=str(task.render_backend),
            psf_jaxes_native=psf_jaxes_native,
        )

        # Per-iter log callback so the run.log shows S2/S3 step magnitudes
        # and convergence per grid (mirrors CPU's "opt iter NN dlnp=..." lines).
        grid_tag = f"grid {task.grid_id}: "
        def _log(msg: str, _tag: str = grid_tag) -> None:
            logger.info(_tag + msg)

        res = solve_grid_3stage(
            multi,
            bg_poly_order=int(task.bg_poly_order),
            single_stage=bool(task.single_stage),
            n_opt_iters=int(task.n_opt_iters),
            pos_step_max_pix=float(task.pos_step_max_pix),
            pos_dlnp_tol_pix=float(task.dlnp_stop_pix),
            position_solver=str(task.position_solver),
            joint_cg_max_iters=int(task.joint_cg_max_iters),
            joint_cg_tol=float(task.joint_cg_tol),
            joint_use_preconditioner=bool(task.joint_use_preconditioner),
            pos_stop_unconverged_frac=float(task.pos_stop_unconverged_frac),
            chi2_ftol=float(getattr(task, 'chi2_ftol', 0.1)),
            fit_sersic_n=bool(getattr(task, 'fit_sersic_n', True)),
            n_fit_min_snr=float(getattr(task, 'n_fit_min_snr', 30.0)),
            wing_peeling=bool(getattr(task, 'wing_peeling', False)),
            wing_peel_tier=int(getattr(task, 'wing_peel_tier', 221)),
            wing_peel_max_sources=int(getattr(task, 'wing_peel_max_sources', 64)),
            blank_response_correction=bool(
                getattr(task, 'blank_response_correction', False)),
            log_fn=_log,
        )
        # Post-solve error measurements (read-only on `multi`/`res`). A
        # failure here fails the grid (R12): the fluxes would ship with errors
        # missing their confusion term. One residual render serves #27 flux
        # confusion + LEG C geometric residual F_corr.
        fluxerr_conf = None
        fcorr_resid = None
        want_conf = bool(getattr(task, "compute_confusion", False))
        want_fres = bool(getattr(task, "compute_fcorr_resid", False))
        if want_conf or want_fres:
            try:
                from .error_model import compute_grid_confusion
                fluxerr_conf, _extra = compute_grid_confusion(
                    multi, res,
                    wing_peeling=bool(getattr(task, "wing_peeling", False)),
                    wing_peel_tier=int(getattr(task, "wing_peel_tier", 221)),
                    wing_peel_max_sources=int(
                        getattr(task, "wing_peel_max_sources", 64)),
                    want_confusion=want_conf,
                    want_fcorr_resid=want_fres,
                    n_grid_slots=int(getattr(task, "n_grid_slots", 1) or 1),
                    log_fn=_log,
                )
                fcorr_resid = _extra.get("fcorr_resid") or None
            except Exception as conf_exc:
                # R12: the grid fails (and is retried with exclusive memory
                # when this was an OOM). The old warning path shipped
                # FLUXERR_CONF NaN, i.e. FLUXERR_TOT without its confusion
                # term (x~2.4 too small on crowded fields), under exit 0.
                raise RuntimeError(
                    f"error model failed: {type(conf_exc).__name__}: "
                    f"{conf_exc}") from conf_exc

        owned_mask = task.owned_source_mask
        if owned_mask is None:
            owned_mask = np.ones(task.pos_x_init.shape[0], dtype=bool)
        else:
            owned_mask = np.asarray(owned_mask, dtype=bool)
            if owned_mask.shape != task.pos_x_init.shape:
                raise ValueError(
                    f"owned_source_mask shape {owned_mask.shape} does not match "
                    f"n_sources={task.pos_x_init.shape[0]}"
                )

        out = PerGridResult(
            grid_id=task.grid_id,
            n_sources=int(np.count_nonzero(owned_mask)),
            pos_x=res.pos_x[owned_mask], pos_y=res.pos_y[owned_mask],
            logre=res.logre[owned_mask], ee1=res.ee1[owned_mask], ee2=res.ee2[owned_mask],
            fluxes=res.fluxes[owned_mask], bg_coefs=res.bg_coefs,
            flux_var=res.flux_var[owned_mask], bg_var=res.bg_var,
            pos_var=res.pos_var[owned_mask],
            source_indices=task.source_indices[owned_mask],
            s1_converged=res.s1_converged,
            s2_converged=res.s2_converged,
            s3_converged=res.s3_converged,
            n_outer_iters=res.n_outer_iters,
            sec_total=time.time() - t0,
            position_stranded=(
                np.asarray(res.position_stranded, dtype=bool)[owned_mask]
                if res.position_stranded is not None else None
            ),
            sersic_n_fit=(
                np.asarray(res.sersic_n_fit, dtype=np.float32)[owned_mask]
                if getattr(res, "sersic_n_fit", None) is not None else None
            ),
            sersic_n_frozen=(
                np.asarray(res.sersic_n_frozen, dtype=bool)[owned_mask]
                if getattr(res, "sersic_n_frozen", None) is not None else None
            ),
            shape_frozen=(
                np.asarray(res.shape_frozen, dtype=bool)[owned_mask]
                if getattr(res, "shape_frozen", None) is not None else None
            ),
            sersic_n_err=(
                np.asarray(res.sersic_n_err, dtype=np.float32)[owned_mask]
                if getattr(res, "sersic_n_err", None) is not None else None
            ),
            fluxerr_conf=(
                fluxerr_conf[owned_mask] if fluxerr_conf is not None else None
            ),
            fcorr_resid=fcorr_resid,   # per-grid per-band (not per-source)
            blank_response_bhat=(
                np.asarray(res.blank_response_bhat, dtype=np.float64)
                if getattr(res, "blank_response_bhat", None) is not None
                else None
            ),
        )
        for shm in attached_shm:
            shm.close()
        return out
    except Exception as exc:
        for shm in attached_shm:
            try:
                shm.close()
            except Exception:
                pass
        import traceback
        tb = traceback.format_exc()
        logger.warning("Grid %d (row=%d col=%d) FAILED: %s\n%s",
                       task.grid_id, task.grid_row, task.grid_col, exc, tb)
        from .bg_polynomial import n_terms_for_order
        M = n_terms_for_order(int(task.bg_poly_order))
        return PerGridResult(
            grid_id=task.grid_id,
            n_sources=int(task.pos_x_init.shape[0]),
            pos_x=task.pos_x_init.copy(),
            pos_y=task.pos_y_init.copy(),
            logre=task.logre_init.copy(),
            ee1=task.ee1_init.copy(),
            ee2=task.ee2_init.copy(),
            fluxes=np.full((task.pos_x_init.shape[0], len(task.bandnames)), np.nan, dtype=np.float32),
            bg_coefs=np.full((len(task.bandnames), M), np.nan, dtype=np.float32),
            flux_var=np.full((task.pos_x_init.shape[0], len(task.bandnames)), np.nan, dtype=np.float32),
            bg_var=np.full((len(task.bandnames), M), np.nan, dtype=np.float32),
            pos_var=np.full((task.pos_x_init.shape[0], 2), np.nan, dtype=np.float32),
            source_indices=task.source_indices,
            s1_converged=False, s2_converged=False, s3_converged=False,
            n_outer_iters=0,
            sec_total=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Worker host-memory management.
#
# Each pool worker is LONG-LIVED and processes many grids sequentially. On
# fields whose grids all have DISTINCT shape signatures (T08285 + extended:
# every grid has a different n_sources, 1726-1818), every shape-keyed
# compiled program (S1 sparse solver, star/mixed Newton kernels, joint
# steps) is compiled fresh per grid and retained forever -- both in the
# module-level shape-keyed dicts and in JAX's global jit caches -- so
# worker host RSS grows deterministically by O(1 GB)/grid. Measured
# consequence: 4 workers x ~11 grids drove the 125 GiB host to 103 GiB
# used / 117% commit and the kernel OOM-killed a worker at the very end
# of the run (BrokenProcessPool at result collection), identically on
# two consecutive runs.
#
# Guard: probe VmRSS after every grid (one log line each) and, above a
# threshold, drop the shape-keyed caches + JAX's compilation caches +
# gc.collect(). Fields whose grids share shapes (the production
# max_sources_per_grid chunking) never cross the threshold, so their
# compile amortization is untouched; distinct-shape fields recompile
# per grid anyway (the caches never hit), so clearing costs nothing but
# frees the dead programs. The persistent JIT disk cache further blunts
# any recompile cost. TRACT7DT_WORKER_RSS_TRIM_GIB tunes the threshold
# (<=0 disables trimming; the probe line is always logged).
# ---------------------------------------------------------------------------

_WORKER_RSS_TRIM_GIB = float(os.environ.get("TRACT7DT_WORKER_RSS_TRIM_GIB", "12"))


def _worker_rss_gib() -> float:
    """Current process VmRSS in GiB (NaN if /proc unavailable)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 ** 2)   # kB -> GiB
    except Exception:
        pass
    return float("nan")


def _trim_worker_host_memory(grid_id: int) -> None:
    """Log worker RSS for this grid; purge JIT caches above the threshold."""
    try:
        rss0 = _worker_rss_gib()
        if not (rss0 == rss0):   # NaN -> no /proc; nothing to do
            return
        if _WORKER_RSS_TRIM_GIB > 0 and rss0 > _WORKER_RSS_TRIM_GIB:
            import gc
            import jax
            from . import forced_phot_sparse as _fps
            from . import solver_grid as _sg
            _fps._SOLVER_CACHE.clear()
            _fps._BATCHED_SOLVER_CACHE.clear()
            _sg._LOCAL_POSITION_STEP_CACHE.clear()
            _sg._JOINT_STEP_JIT_CACHE.clear()
            _sg._VARPRO_STEP_JIT_CACHE.clear()
            _sg._VARPRO_NERR_JIT_CACHE.clear()
            jax.clear_caches()
            gc.collect()
            rss1 = _worker_rss_gib()
            logger.info(
                "grid %d: worker RSS %.1f GiB > %.0f GiB threshold -> "
                "cleared shape-keyed JIT caches (RSS now %.1f GiB)",
                grid_id, rss0, _WORKER_RSS_TRIM_GIB, rss1,
            )
        else:
            logger.info("grid %d: worker RSS %.1f GiB", grid_id, rss0)
    except Exception as exc:   # never fail a grid over memory bookkeeping
        logger.warning("grid %d: worker memory trim failed: %s", grid_id, exc)


def run_one_grid_managed(task: PerGridTask) -> PerGridResult:
    """Pool entry: fit one grid, then probe/trim worker host memory.

    The trim runs AFTER :func:`run_one_grid` returns (its frame -- and the
    multi-GB per-grid arrays it referenced -- are collectable by then) and
    only ever drops caches, so the OOM-retry and shm machinery are
    untouched. The returned result is plain numpy.
    """
    out = run_one_grid(task)
    _trim_worker_host_memory(task.grid_id)
    return out


__all__ = [
    "PerGridTask",
    "PerGridResult",
    "SharedArraySlice",
    "_worker_init",
    "_worker_ping",
    "run_one_grid",
    "run_one_grid_managed",
]
