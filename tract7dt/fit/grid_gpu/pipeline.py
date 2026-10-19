"""Top-level entry point for the grid-based GPU image fit.

Dispatched from :func:`tract7dt.pipeline.run_pipeline` (``compute.mode: gpu``,
the only mode since card 018 removed the per-patch CPU/Tractor flow).
Output is written to ``output_catalog.csv`` for the ZP calibration stage.
"""

from __future__ import annotations

import atexit
import logging
import multiprocessing as mp
import functools
import os
import signal
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ... import run_status
from ...run_status import RunContractError
from ...utils import case_insensitive_col_map
from .error_columns import add_error_columns
from .grid import (
    GridDef, assign_sources_to_grids, divide_image_into_grids, slice_band_image,
)
# Host-memory probes live in error_model, next to the estimators whose
# footprint they size (T2, 2026-08-03). They were written here first, for the
# raw-sky pass (T1); a second copy there would be the same rule written twice,
# so both callers share one definition and the old names still resolve here.
#   _host_mem_available_bytes -- MemAvailable, already net of this process.
#   _c_alloc_in_use_bytes     -- glibc mallinfo2 in-USE bytes, NOT RSS (the
#     raw-sky pass runs right after the fit releases GBs of host arrays, so the
#     allocator sits on a warm arena and an RSS delta reads ~0).
from .error_model import _c_alloc_in_use_bytes, _host_mem_available_bytes
from .psf_polynomial import PsfPolynomial
from .runner import (
    PerGridResult, PerGridTask, SharedArraySlice,
    _worker_init, _worker_ping, run_one_grid, run_one_grid_managed,
)
from .shm_hygiene import sweep_stale_segments
from .band_flags import (
    PARTIAL_COVERAGE_MAX, PSF_LOBE_FRAC, blank_nodata_band_cells,
    compose_band_flags, psf_lobe_zone, psf_uncovered_fraction)
from .source_model import (
    SOURCE_DEV,
    SOURCE_EXP,
    SOURCE_SERSIC,
    SOURCE_STAR,
    catalog_source_params,
    ellipse_soft_to_re_ab_phi_np,
    is_supported_source_type,
)


logger = logging.getLogger("tract7dt.fit.grid_gpu.pipeline")


def _resolve_gpu_devices(
    devices_cfg: Any, n_grids_needed: int,
) -> list[int]:
    """Resolve ``compute.gpu.devices`` (or ``"auto"``) into a concrete
    device list. Capped at ``n_grids_needed`` so we don't reserve more
    GPUs than grids."""
    if devices_cfg == "auto" or devices_cfg is None:
        try:
            import jax
            devices = [int(d.id) for d in jax.devices("cuda")]
        except Exception as exc:
            raise SystemExit(
                f"compute.mode: gpu requires CUDA-enabled jaxlib + "
                f">= 1 visible CUDA device; jax.devices('cuda') failed: {exc}"
            )
    elif isinstance(devices_cfg, list):
        devices = [int(x) for x in devices_cfg]
    else:
        raise SystemExit(
            f"compute.gpu.devices must be 'auto' or a list of int; got {devices_cfg!r}"
        )
    if not devices:
        raise SystemExit("compute.mode: gpu but no CUDA devices visible to JAX.")
    return devices[:n_grids_needed]


def _gpu_free_mib(device_ids: list[int]) -> dict[int, int]:
    """Per-device FREE GPU memory (MiB) via ``nvidia-smi``. Machine-adaptive
    (reads the actual card size). On any failure returns {} and the caller
    falls back to a conservative default."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001 - nvidia-smi missing / odd driver
        logger.warning("grid_gpu: nvidia-smi free-mem query failed (%s); "
                       "adaptive packing assumes a conservative default", exc)
        return {}
    free: dict[int, int] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                free[int(parts[0])] = int(float(parts[1]))
            except ValueError:
                continue
    return free


def _grid_task_dims(task: Any) -> "tuple[int, int, int, int, tuple[int, int] | None]":
    """``(n_src, n_bands, psf_h, psf_w, (band_h, band_w) | None)`` read off a
    :class:`PerGridTask` -- the single source of the shapes for BOTH the
    structural peak model and the measured probe (one derivation, so the two
    paths cannot silently read different geometry). ``None`` band geometry
    means the task carries neither shared-slice specs nor inline arrays."""
    n_src = int(getattr(task, "source_indices").size)
    n_bands = max(1, len(task.bandnames))
    ph, pw = task.psf_shape if getattr(task, "psf_shape", None) else (55, 55)
    band_hw = None
    try:
        if task.shared_data_per_band:
            s = task.shared_data_per_band[0]
            band_hw = (int(s.y1 - s.y0), int(s.x1 - s.x0))
        elif task.data_per_band:
            d0 = task.data_per_band[0]
            band_hw = (int(d0.shape[0]), int(d0.shape[1]))
    except Exception:  # noqa: BLE001
        band_hw = None
    return n_src, n_bands, int(ph), int(pw), band_hw


def _estimate_grid_peak_mib(task: Any) -> float:
    """STRUCTURAL FALLBACK estimate of ONE grid's PEAK GPU memory (MiB).

    G1 (2026-08-18): the pack decision now uses the MEASURED per-grid peak
    (:func:`_measured_grid_peak_mib` -- XLA's own per-call report + residency
    arithmetic, the T5 template). This hand-calibrated model remains ONLY as
    the fallback for the paths the probe does not cover: 3-stage
    (non-``single_stage``) tasks, whose S2/S3 position-step working sets are
    real and unprobed (the ``95.0`` term territory), and any environment where
    the probe cannot run (no CUDA/JAX in the parent, lowering failure). On
    S1-only island tasks it measured 3.2x (sparse) / 11.2x (crowded) ABOVE
    the real demand (development ledger SS19-4) because the position-step term
    prices code that never runs there -- which is exactly why the measured
    path replaced it as the primary. Consumers must pair it with
    ``PEAK_HEADROOM`` (the pairing it was calibrated under).

    Dominant allocations in the 3-stage solve: the per-source position-step
    working set (~n_src x n_bands x psf_area, inflated by the autodiff Hessian
    + line-search re-renders) plus the resident per-band image/background
    stacks (~n_bands x grid_pixels). Calibrated so a ~1750-source, 20-band,
    55x55-PSF grid on a 5800x9200 image is ~11.5 GiB -- slightly ABOVE the
    observed ~10 GiB OOM threshold, i.e. deliberately conservative. Only sets
    the INITIAL worker count; the BFC per-worker cap + OOM-retry make any
    estimate error safe (never corrupts output)."""
    n_src, n_bands, ph, pw, band_hw = _grid_task_dims(task)
    psf_area = ph * pw
    grid_pix = band_hw[0] * band_hw[1] if band_hw is not None else 1_250_000
    base = 1.0e9
    # The position-step vmap is CHUNKED (TRACT7DT_POS_STEP_CHUNK, default 512,
    # within-noise) -> only CHUNK sources' working set is live at once, so the
    # per-source term is bounded by the chunk size (this is what lets large-grid
    # fields pack workers/GPU). 0 disables chunking (full per-source term,
    # bit-exact solver). Keep in sync with solver_grid._POS_STEP_CHUNK.
    chunk = int(os.environ.get("TRACT7DT_POS_STEP_CHUNK", "512"))
    eff_src = n_src if chunk <= 0 else min(n_src, chunk)
    src_term = 95.0 * eff_src * n_bands * psf_area    # chunked per-source working set
    img_term = 16.0 * n_bands * grid_pix              # resident image/bg stacks
    return (base + src_term + img_term) / (1024.0 ** 2)


# Keep in sync with the `window = 8` literal in solver_grid.stage1_solve's
# launch/collect loop (the bounded in-flight band window). The residency
# arithmetic below charges window+1 stamp buffers because the loop collects
# only AFTER the (window+1)-th launch.
_GRID_SOLVE_WINDOW = 8


def _grid_resident_bytes(task: Any) -> "tuple[int, dict] | None":
    """Device bytes RESIDENT across one S1-only grid solve -- pure arithmetic
    from the task's real shapes and the engine's actual dtypes, no tuned
    constants. Terms (each anchored to the code that allocates it):

    - every band's data/invvar/pix_good uploaded up front at
      ``GridBandInputs`` construction (f32 + f32 + bool = 9 B/px/band);
    - ONE shared bg basis, ``grid_pix x M`` f32 (``runner.run_one_grid`` ->
      ``make_bg_basis_jax``), M from ``bg_poly_order``;
    - the in-flight per-band stamp window of ``stage1_solve``
      (``(window+1) x n_src x psf_area`` f32);
    - neighbor list/mask (n x 32 x 5 B) + per-source parameter columns and
      per-band solve outputs (small, kept exact rather than dropped);
    - one full-band model plane when the post-solve residual pass will run
      (``compute_confusion`` / ``compute_fcorr_resid`` -- the eager render +
      scatter in ``compute_grid_confusion``).

    Returns ``(bytes, detail)`` or None when the band geometry is unknown
    (the probe is then unusable and the caller falls back)."""
    from .bg_polynomial import n_terms_for_order
    from .forced_phot_sparse import DEFAULT_K_MAX

    n_src, n_bands, ph, pw, band_hw = _grid_task_dims(task)
    if band_hw is None:
        return None
    pix = band_hw[0] * band_hw[1]
    M = int(n_terms_for_order(int(task.bg_poly_order)))
    psf_area = ph * pw
    b_bands = n_bands * pix * 9                       # f32 data + f32 invvar + bool good
    b_bg = pix * M * 4                                # shared bg basis, f32
    b_stamps = (_GRID_SOLVE_WINDOW + 1) * n_src * psf_area * 4
    b_small = (n_src * (DEFAULT_K_MAX * 5 + 64)      # neighbor idx i32 + mask bool; param cols
               + n_bands * (n_src + M) * 8)          # per-band flux/var outputs
    postsolve = bool(getattr(task, "compute_confusion", False)
                     or getattr(task, "compute_fcorr_resid", False))
    b_model = pix * 4 if postsolve else 0             # eager model plane, f32
    total = b_bands + b_bg + b_stamps + b_small + b_model
    detail = {"bands": b_bands, "bg": b_bg, "stamps": b_stamps,
              "small": b_small, "model": b_model, "M": M, "pix": pix}
    return total, detail


def _grid_s1_call_bytes(task: Any) -> "int | None":
    """XLA's own report (``temp + argument + output``) for ONE per-band S1
    solve call at this task's EXACT compiled shapes (the same meter as the
    island path's ``_xla_call_bytes``, validated there to 0.6% of NVML).

    Compiled with PRODUCTION options (autotuner on) so the executable lands
    in the persistent JIT cache and the worker that fits this grid REUSES it
    -- the probe compile is the production compile done early, not a sunk
    cost (contrast the island width probes' no-autotune executables, development
    ledger SS17). Bonus: that worker then never runs the autotuner for
    this shape, so its BFC cap only has to hold the steady state.

    Lowered under ``jax.enable_x64()`` to match the worker runtime
    (``_worker_init`` sets ``JAX_ENABLE_X64=1``; the kernel solves its normal
    equations in f64 -- lowering with x64 off would silently price a
    DIFFERENT all-f32 program). None on any failure."""
    try:
        import jax
        import jax.numpy as jnp

        from .bg_polynomial import n_terms_for_order
        from .forced_phot_sparse import DEFAULT_K_MAX, _make_solve_band_jit

        n_src, n_bands, ph, pw, band_hw = _grid_task_dims(task)
        if band_hw is None or n_src <= 0:
            return None
        bh, bw = band_hw
        M = int(n_terms_for_order(int(task.bg_poly_order)))
        k = int(DEFAULT_K_MAX)
        with jax.enable_x64():
            S = jax.ShapeDtypeStruct
            f32, i32, b1 = jnp.float32, jnp.int32, jnp.bool_
            args = (
                S((bh, bw), f32), S((bh, bw), f32), S((bh, bw), b1),
                S((n_src, ph, pw), f32), S((n_src,), i32), S((n_src,), i32),
                S((n_src, k), i32), S((n_src, k), b1),
                S((bh * bw, M), f32), S((), f32),
            )
            _, fn = _make_solve_band_jit(ph, pw, bh, bw, n_src, k, M)
            comp = fn.lower(*args).compile()
            m = comp.memory_analysis()
        if m is None:
            return None
        return int(m.temp_size_in_bytes + m.argument_size_in_bytes
                   + m.output_size_in_bytes)
    except Exception:  # noqa: BLE001
        return None


def _parent_backend_is_gpu() -> bool:
    """True iff THIS process's default JAX backend is a GPU. The probe's
    numbers are only honest for the backend the workers will use; with an
    explicit ``compute.gpu.devices`` list the parent may reach the chooser
    without CUDA ever having initialized, and JAX silently falls back to CPU
    -- CPU-backend memory reports must not be adopted as the GPU peak.
    (Unit tests monkeypatch this: the CPU-backend probe is still the right
    plumbing check there.)"""
    try:
        import jax
        return str(jax.default_backend()).lower() == "gpu"
    except Exception:  # noqa: BLE001
        return False


def _measured_grid_peak_mib(
    tasks: list, gpu_cfg: dict,
) -> "tuple[dict | None, str]":
    """MEASURED per-grid peak for the pack decision (G1, the T5 template):
    budget stays ``SAFETY x`` the card's probed free memory; cost = residency
    arithmetic (:func:`_grid_resident_bytes`) + the LARGER of the two probed
    per-call XLA reports -- the S1 band-solve kernel and, when the final-pass
    flags are on, the post-solve confusion kernel. Measured cost against
    measured budget, no headroom constant; the residual risk (probe blind
    spots such as the eager render transients) is carried by the SAME safety
    net the old model relied on for its own misestimates: the per-worker BFC
    cap + the exclusive-GPU OOM retry.

    Probing every task would compile every shape, so a derivation-only
    surrogate RANKS the tasks and only the top candidates are probed (the T5
    rule: the surrogate only ever chooses what to MEASURE; no unmeasured
    number is adopted). Candidates = argmax by surrogate, by n_src, and by
    grid pixels (usually one and the same task).

    Returns ``(info, diagnostic)`` where ``info`` is a dict with
    ``peak_mib`` (resident + max kernel call), ``resident_mib``,
    ``kernel_mib`` and ``arena_mib`` (the per-worker arena requirement,
    ``resident + (window+1) x kernel``, maximised over the probed
    candidates -- the ARENA maximand, which is not the same ordering as the
    peak), or ``(None, reason)`` when the probe cannot run -- the caller
    then falls back to the structural model with PEAK_HEADROOM."""
    if not tasks:
        return None, "no tasks"
    if not all(bool(getattr(t, "single_stage", False)) for t in tasks):
        return None, "3-stage tasks (S2/S3 working sets unprobed)"
    if not _parent_backend_is_gpu():
        return None, "parent JAX backend is not GPU (probe would price CPU)"
    residents: list = []
    for t in tasks:
        r = _grid_resident_bytes(t)
        if r is None:
            return None, "band geometry unknown"
        residents.append(r)

    def _surrogate(i: int) -> float:
        # Ranking ONLY (never adopted): resident + the S1 kernel's dominant
        # temp scales -- the (n,psf,psf,M) bg gather and the (n+M)^2 f32+f64
        # normal-equation blocks.
        n_src, _nb, ph, pw, _hw = _grid_task_dims(tasks[i])
        M = residents[i][1]["M"]
        return (residents[i][0]
                + n_src * ph * pw * (2 + M) * 4.0
                + 3.0 * (n_src + M) * (n_src + M) * 8.0)

    cand = {max(range(len(tasks)), key=_surrogate),
            max(range(len(tasks)),
                key=lambda i: _grid_task_dims(tasks[i])[0]),
            max(range(len(tasks)), key=lambda i: residents[i][1]["pix"])}
    # Parent-side persistent cache: the probe compiles must be the SAME cache
    # entries the workers load (runner.enable_persistent_jit_cache is the one
    # implementation both use).
    try:
        from .runner import enable_persistent_jit_cache
        enable_persistent_jit_cache(gpu_cfg.get("jit_cache_dir", "auto"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("grid_gpu: probe cache setup failed (%s); probe "
                       "compiles will not persist", exc)
    best: "tuple[float, dict] | None" = None
    for i in sorted(cand):
        t = tasks[i]
        s1 = _grid_s1_call_bytes(t)
        if s1 is None:
            return None, "S1 kernel probe failed (lowering/compile)"
        conf = 0
        conf_failed = False
        if bool(getattr(t, "compute_confusion", False)):
            from .error_model import confusion_call_bytes
            n_src, _nb, ph, pw, hw = _grid_task_dims(t)
            c = confusion_call_bytes(hw[0], hw[1], ph, pw, n_src)
            if c is None:
                conf_failed = True     # keep the validated S1 composition
            else:
                conf = int(c)
        kernel = max(s1, conf)
        # Rank candidates by the ARENA requirement -- the number the caller
        # actually provisions per worker -- not by the plain peak: a
        # kernel-heavy candidate can need a bigger arena than a
        # resident-heavy one with the larger resident+kernel sum.
        need = residents[i][0] + (_GRID_SOLVE_WINDOW + 1) * kernel
        if best is None or need > best[0]:
            best = (need, {"i": i, "s1": s1, "conf": conf,
                           "conf_failed": conf_failed,
                           "resident": residents[i][0], "kernel": kernel})
    assert best is not None
    need, d = best
    n_src, n_bands, _ph, _pw, _hw = _grid_task_dims(tasks[d["i"]])
    mib = 1024.0 ** 2
    diag = (f"S1 {d['s1'] / mib:.0f} + conf {d['conf'] / mib:.0f} + "
            f"resident {d['resident'] / mib:.0f} MiB at n={n_src}, "
            f"{n_bands} band(s), {len(cand)} shape(s) probed"
            + ("; conf probe failed" if d["conf_failed"] else ""))
    info = {
        "peak_mib": (d["resident"] + d["kernel"]) / mib,
        "resident_mib": d["resident"] / mib,
        "kernel_mib": d["kernel"] / mib,
        "arena_mib": need / mib,
    }
    return info, diag


def _physical_gpu_index(device_id: int) -> int:
    """Map a jax-visible device id to its PHYSICAL nvidia-smi index through
    the parent's ``CUDA_VISIBLE_DEVICES`` restriction -- the same mapping the
    2026-08-15 worker-pinning fix applies (``runner._worker_init``): with no
    restriction this is the identity. ``nvidia-smi`` ignores
    ``CUDA_VISIBLE_DEVICES``, so indexing its output by the jax id reads the
    wrong card whenever the parent runs restricted."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return int(device_id)
    vis = [x.strip() for x in cvd.split(",") if x.strip()]
    try:
        return int(vis[int(device_id)])
    except (IndexError, ValueError):
        return int(device_id)


def _gpu_free_total_mib(device_ids: list[int]) -> "dict[int, tuple[int, int]]":
    """Per-device (FREE, TOTAL) GPU memory (MiB) via ``nvidia-smi``, keyed by
    the caller's JAX-VISIBLE device ids (physical mapping via
    :func:`_physical_gpu_index`). Empty on failure (caller falls back). The
    measured pack decision needs BOTH: free is the budget, total is what a
    worker's ``XLA_PYTHON_CLIENT_MEM_FRACTION`` is a fraction of.

    NOTE: the legacy `_gpu_free_mib` (structural-fallback path) keeps its
    historical physical-index keying untouched -- registered gap, see the development
    ledger SS20."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        logger.warning("grid_gpu: nvidia-smi free/total query failed (%s)", exc)
        return {}
    phys: dict[int, tuple[int, int]] = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                phys[int(parts[0])] = (int(float(parts[1])),
                                       int(float(parts[2])))
            except ValueError:
                continue
    mem: dict[int, tuple[int, int]] = {}
    for d in device_ids:
        got = phys.get(_physical_gpu_index(int(d)))
        if got is not None:
            mem[int(d)] = got
    return mem


def _parent_cuda_overhead_mib() -> "float | None":
    """THIS process's device-memory footprint (MiB), read from the driver
    (``nvidia-smi --query-compute-apps``).

    At chooser time the parent's XLA buffers are freed, so what remains is
    ~the fixed per-process cost of a CUDA context plus the loaded kernel
    modules for this engine's library stack -- the same stack every grid
    worker loads. It therefore prices the per-worker NON-ARENA overhead
    (CUBIN module loads, cuBLAS handles, CUDA graphs -- none of it lives in
    the XLA arena) that the 2026-08-18 first G1 build left out: with 90% of
    the card preallocated as equal-split arenas, 133/148 crowded grids died
    at CUBIN-load/cublas-init before touching their arenas. Measured on this
    machine/driver/library stack, not assumed; under MPS the workers share
    context state, so the parent's direct-context number errs conservative.
    None when the parent is not a CUDA process or the query fails."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
        me = os.getpid()
        vals = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == me:
                vals.append(float(parts[1]))
        if not vals:
            return None
        return max(vals)
    except Exception:  # noqa: BLE001
        return None


def _catalog_summary(out: "pd.DataFrame", bands: list) -> dict:
    """RUN_STATUS.json summary of the GPU catalog: rows, fitted rows,
    reliable rows, and per band the fraction of fitted rows with a finite flux
    (NaN = no data or saturated, R05/R11)."""
    fitted = pd.to_numeric(out.get("x_pix_white_fit"), errors="coerce").notna()
    n_fit = int(fitted.sum())
    summ = {"rows": int(len(out)), "fitted_rows": n_fit,
            "reliable_rows": int(out["fit_is_reliable"].fillna(False)
                                 .astype(bool).sum())
            if "fit_is_reliable" in out.columns else None,
            "finite_flux_fraction": {}}
    for b in bands:
        col = f"FLUX_{b}_fit"
        if col in out.columns and n_fit:
            v = pd.to_numeric(out.loc[fitted, col], errors="coerce")
            summ["finite_flux_fraction"][b] = round(
                float(np.isfinite(v.to_numpy(dtype=float)).mean()), 4)
    return summ


def _raise_if_grids_failed(results: list) -> None:
    """R12: end the run when any grid result still carries an error."""
    failed = [r for r in results if getattr(r, "error", None)]
    if not failed:
        return
    logger.error("grid_gpu: %d grid(s) FAILED even after retry: %s",
                 len(failed), [(r.grid_id, str(r.error)[:120]) for r in failed[:5]])
    raise RunContractError(
        "grid_failed",
        f"{len(failed)} of {len(results)} grid(s) failed after the retry; the "
        "catalog would carry their sources as NaN rows",
        grids=[{"grid_id": int(r.grid_id), "error": str(r.error)[:500]}
               for r in failed])


def _is_oom_error(res: Any) -> bool:
    """True if a per-grid result failed with a GPU out-of-memory error."""
    err = getattr(res, "error", None)
    if not err:
        return False
    e = str(err).upper()
    return ("RESOURCE_EXHAUSTED" in e) or ("OUT OF MEMORY" in e) or ("OOM" in e)


_FCORR_BYTES_PER_PIXEL: "float | None" = None


def _fcorr_bytes_per_pixel() -> tuple[float, str]:
    """Host bytes ``measure_correlated_noise_factor`` holds per pixel of its
    working crop -- MEASURED, once per process, by running the estimator on a
    small synthetic sky while sampling the allocator.

    Replaces a hand-counted "about eleven image-sized float64 planes". That
    count was not tuned to our data, but it was still a number a human read out
    of the code, so it drifts silently the moment the estimator allocates one
    more array -- and it only ever existed to pick a thread count. Measuring the
    real thing costs ~0.3 s once and re-measures itself on any machine, any
    numpy, any future version of the estimator.

    Falls back to the structural count if the allocator cannot be read (non-glibc)
    or the measurement comes back empty; the reason is returned for the log.
    """
    global _FCORR_BYTES_PER_PIXEL
    if _FCORR_BYTES_PER_PIXEL is not None:
        return _FCORR_BYTES_PER_PIXEL, "cached"
    _STRUCTURAL = 11 * 8.0     # fallback only: ~11 image-sized float64 planes
    base = _c_alloc_in_use_bytes()
    if base <= 0:
        _FCORR_BYTES_PER_PIXEL = _STRUCTURAL
        return _STRUCTURAL, "structural (allocator not readable)"
    try:
        from .error_model import measure_correlated_noise_factor

        n = 1024
        rng = np.random.default_rng(0)
        img = rng.normal(0.0, 10.0, (n, n)).astype(np.float32)
        yy, xx = np.mgrid[-13:14, -13:14]
        psf = np.exp(-(yy ** 2 + xx ** 2) / (2 * 2.1 ** 2)).astype(np.float32)
        psf /= psf.sum()
        peak = [base]
        stop = threading.Event()

        def _sample():
            while not stop.is_set():
                peak[0] = max(peak[0], _c_alloc_in_use_bytes())
                stop.wait(0.002)

        th = threading.Thread(target=_sample, daemon=True)
        th.start()
        try:
            _F, dg = measure_correlated_noise_factor(img, None, psf,
                                                     band="_probe")
        finally:
            stop.set()
            th.join(timeout=1.0)
        # the probe image + PSF are ours, not the estimator's working set
        own = img.nbytes + psf.nbytes
        bpp = max(0.0, (peak[0] - base - own)) / float(n * n)
        if not np.isfinite(bpp) or bpp <= 0 or dg.get("fallback"):
            _FCORR_BYTES_PER_PIXEL = _STRUCTURAL
            return _STRUCTURAL, (
                f"structural (probe unusable: fallback={dg.get('fallback')!r}, "
                f"bpp={bpp:.1f})")
        _FCORR_BYTES_PER_PIXEL = float(bpp)
        return float(bpp), f"measured on a {n}x{n} synthetic sky"
    except Exception as exc:                                   # noqa: BLE001
        _FCORR_BYTES_PER_PIXEL = _STRUCTURAL
        return _STRUCTURAL, f"structural (probe failed: {type(exc).__name__})"


def _fcorr_band_workers(bands: list, image_dict: dict) -> tuple[int, str]:
    """Threads for the raw-sky F_corr band pass. Measured, not tuned.

    Bounded by three things, all measured at the call site:
      (a) work        -- never more threads than bands;
      (b) cores       -- os.cpu_count();
      (c) host memory -- ONE band's peak = the estimator's MEASURED bytes per
          working-crop pixel (:func:`_fcorr_bytes_per_pixel`) times that crop's
          area, ``ch, cw = min(shape, _FCORR_MAX_CROP)``. Divide MemAvailable
          by that.

    Same probe-and-divide shape as :func:`_choose_workers_per_gpu`, and it
    self-limits: MemAvailable already excludes the pipeline's own live arrays,
    so no headroom constant is needed here.
    """
    from .error_model import _FCORR_MAX_CROP

    bpp, how = _fcorr_bytes_per_pixel()
    peak = 0
    for bn in bands:
        arr = (image_dict.get(bn) or {}).get("img_scaled")
        if arr is None:
            continue
        h, w = np.asarray(arr).shape
        ch = min(int(h), _FCORR_MAX_CROP)
        cw = min(int(w), _FCORR_MAX_CROP)
        peak = max(peak, int(bpp * ch * cw))
    if not bands or peak <= 0:
        return 1, "no bands"
    avail = _host_mem_available_bytes()
    by_mem = max(1, avail // peak) if avail > 0 else 1
    n = int(max(1, min(len(bands), os.cpu_count() or 1, by_mem)))
    return n, (f"{len(bands)} band(s), {os.cpu_count()} core(s), "
               f"{avail / 2**30:.0f} GiB avail / {peak / 2**30:.2f} GiB per band "
               f"({bpp:.0f} B/pixel, {how}) -> {n} thread(s)")


def _fcorr_raw_sky_pass(
    bands: list, image_dict: dict, psfs: dict,
) -> tuple[dict, list]:
    """Raw-sky correlated-noise factor for every band, BAND-PARALLEL.

    Returns ``(fcorr_raw_by_band, diags)``; ``diags`` is in band order.

    Each band's estimate is an independent, deterministic pure-numpy/scipy
    computation on that band's own image, and the GPUs are idle at this point
    in the run -- the historical serial loop left ~60 cores unused while
    spending ~4.7 s/band (measured 92-114 s per pass on 20-22-band tiles: 4% of
    a crowded run, 23% of a sparse one, and the same cold or warm because
    neither JIT nor the page cache is involved).

    THREADS, not processes: the hot operations (``np.fft.rfft2`` / ``irfft2``
    and ``scipy.signal.fftconvolve``) release the GIL, so a 22-thread pool
    measured 8.7x while a 16-process pool reached only 6.5x after paying to
    pickle ~1.8 GiB of band images.

    Determinism: results are consumed in BAND order (not completion order) and
    the estimator's per-band log lines are buffered and emitted by the calling
    thread, so ``correlated_noise_factors.csv`` and ``run.log`` are unchanged.
    A band that raises is isolated exactly as in the serial version (warn,
    F_corr = 1.0) and never poisons its neighbours.
    """
    from .error_model import measure_correlated_noise_factor
    from .psf_polynomial import evaluate_psf_at_np

    fcorr_raw_by_band: dict[str, float] = {}
    diags: list = []

    def _one(bn: str):
        bd = image_dict[bn]
        img_s = bd["img_scaled"]
        bad_s = bd.get("bad")
        Hs, Ws = np.asarray(img_s).shape
        # central PSF stamp (field centre), pure numpy
        phi = evaluate_psf_at_np(psfs[bn], Ws * 0.5, Hs * 0.5)
        msgs: list[str] = []
        F_corr, dg = measure_correlated_noise_factor(
            img_s, bad_s, phi, band=bn, log_fn=msgs.append)
        return float(F_corr), dg, msgs

    n_workers, why = _fcorr_band_workers(bands, image_dict)
    logger.info("grid_gpu: raw-sky F_corr band pass: %s", why)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [(bn, ex.submit(_one, bn)) for bn in bands]
        for bn, fut in futs:
            try:
                F_corr, dg, msgs = fut.result()
                for m in msgs:
                    logger.info(m)
                fcorr_raw_by_band[bn] = F_corr
                diags.append(dg)
            except Exception as e:  # never let an error pass poison the run
                logger.warning(
                    "grid_gpu: raw-sky F_corr[%s] failed (%s); F_corr=1.0",
                    bn, str(e)[:120])
                fcorr_raw_by_band[bn] = 1.0
    return fcorr_raw_by_band, diags


def _choose_workers_per_gpu(
    tasks: list, devices: list[int], gpu_cfg: dict,
) -> tuple[int, str, float, str]:
    """Adaptively choose workers-per-GPU from FREE GPU memory (probed) + a
    conservative per-grid peak estimate, so packing is safe on ANY machine and
    for ANY image/catalog. Returns (workers_per_gpu, alloc_mode,
    per_worker_mem_fraction, diagnostic).

    Honors an explicit override: env ``TRACT7DT_GPU_WORKERS_PER_GPU`` (wins) or
    ``compute.gpu.workers_per_gpu`` (int); ``"auto"`` (default) computes it.
    When >1 workers/GPU, alloc_mode is ``"bfc"`` so each worker is HARD-capped
    at ~0.9/N of the card (co-located workers cannot collectively OOM); =1 uses
    the on-demand ``platform`` allocator (solo worker owns the GPU)."""
    MAX_WPG = 8
    SAFETY = 0.9
    default_mem = float(gpu_cfg.get("mem_fraction", 0.5))
    override = os.environ.get("TRACT7DT_GPU_WORKERS_PER_GPU")
    if override is None:
        cfg_wpg = gpu_cfg.get("workers_per_gpu", "auto")
        if str(cfg_wpg).strip().lower() not in ("auto", "", "none"):
            override = str(cfg_wpg)
    if override is not None:
        try:
            wpg = max(1, int(override))
        except ValueError:
            wpg = 1
        mode = "bfc" if wpg > 1 else "platform"
        mem = (SAFETY / wpg) if wpg > 1 else default_mem
        return wpg, mode, mem, f"override workers_per_gpu={wpg} ({mode})"
    if not tasks:
        return 1, "platform", default_mem, "no grids"
    # G1 (2026-08-18): the requirement side is MEASURED when the probe can run
    # (S1-only tasks; XLA per-call report + residency arithmetic). Probe FIRST,
    # free-memory query SECOND, so the probe's own compile transients are
    # released before the budget is read. The probe must NEVER kill the run:
    # any escape (e.g. an out-of-range bg_poly_order raising inside the
    # residency arithmetic) demotes to the structural fallback.
    try:
        info, peak_how = _measured_grid_peak_mib(tasks, gpu_cfg)
    except Exception as exc:  # noqa: BLE001
        info, peak_how = None, f"probe raised {type(exc).__name__}: {exc}"
    ovh_mib: "float | None" = None
    total_mib = 0
    if info is not None:
        # Per-worker NON-ARENA driver overhead (context + CUBIN modules +
        # cuBLAS/graphs), measured from the parent's own footprint. Without
        # it the pack decision is blind to exactly what killed the first G1
        # build (133/148 crowded grids dead at CUBIN load under 8w/GPU with
        # equal-split arenas holding 90% of the card).
        ovh_mib = _parent_cuda_overhead_mib()
        mem_ft = _gpu_free_total_mib(devices)
        total_mib = min((mem_ft.get(int(d), (0, 0))[1] for d in devices),
                        default=0)
        if ovh_mib is None or total_mib <= 0:
            peak_how = ("worker overhead/card size unmeasurable "
                        f"(after: {peak_how})")
            info = None
    free = _gpu_free_mib(devices)
    free_mib = min((free.get(int(d), 0) for d in devices), default=0)
    if free_mib <= 0:
        free_mib = 12000  # conservative fallback (~half a 24 GB card) if probe failed
    if info is not None:
        # Measured cost vs measured budget (the T5 form), extended to the
        # ARENA: each packed worker preallocates an arena sized to its
        # measured need -- resident + (window+1) x the largest per-call
        # kernel report -- instead of an equal split of the card. The
        # (window+1) multiplier is the engine's own in-flight band window
        # (stage1_solve launches up to window+1 solves before collecting;
        # it also bounds the autotuner's compile-time transient, which
        # benchmarks one op at a time with candidate workspace <= a call's
        # own footprint). No headroom constant anywhere; the BFC hard cap +
        # exclusive-GPU OOM retry carry the residual risk, and the old
        # collective guarantee (sum of caps <= SAFETY x card) is kept by
        # the min() below.
        arena_mib = float(info["arena_mib"])
        eff_req = arena_mib + ovh_mib
        basis = (f"measured grid peak ~{info['peak_mib']:.0f}MiB ({peak_how}), "
                 f"arena ~{arena_mib:.0f}MiB + worker overhead "
                 f"~{ovh_mib:.0f}MiB")
    else:
        max_peak = max(_estimate_grid_peak_mib(t) for t in tasks)
        # PEAK_HEADROOM (structural-fallback pairing): the BFC arena
        # (PREALLOCATE per-worker cap) needs comfortably MORE than the modeled
        # peak to fit a grid without OOM -- fragmentation + model uncertainty.
        # Empirically (T08285): packing where cap ~= est OOM'd every grid,
        # while cap ~= 2.5x est fit cleanly. The OOM-retry still backstops any
        # residual misestimate. Applies ONLY to the hand-constant model it was
        # calibrated with -- the measured branch above replaces both together.
        PEAK_HEADROOM = 2.5
        arena_mib = 0.0
        eff_req = PEAK_HEADROOM * max_peak if max_peak > 0 else 0.0
        basis = (f"est max grid peak ~{max_peak:.0f}MiB (x{PEAK_HEADROOM} "
                 f"headroom; structural model: {peak_how})")
    wpg = int(SAFETY * free_mib / eff_req) if eff_req > 0 else 1
    if info is not None:
        # Never spawn more worker SLOTS than there are grids: slots =
        # len(devices) x wpg, so the per-GPU cap is ceil(n_tasks / n_devices)
        # (the derived form of the legacy min(wpg, len(tasks)) clip -- that
        # one let a small field spawn devices x tasks idle workers once the
        # requirement side stopped being a 2.5x-inflated model). Applied on
        # the MEASURED branch only: the structural fallback keeps the legacy
        # clip verbatim so unprobed configs (3-stage/legacy-bit-compat)
        # behave exactly as before G1.
        wpg = max(1, min(MAX_WPG, wpg,
                         -(-len(tasks) // max(1, len(devices)))))
    else:
        wpg = max(1, min(MAX_WPG, wpg, len(tasks)))   # legacy clip, verbatim
    mode = "bfc" if wpg > 1 else "platform"
    if wpg <= 1:
        mem = default_mem
    elif info is not None:
        # Need-sized arena as a fraction of the card; the min() preserves the
        # legacy collective guarantee (sum of caps <= SAFETY of the card),
        # and the floor keeps the fraction one %.4f ulp above zero -- the
        # worker env serializes it as f"{mem:.4f}" and a 0.0000 cap would
        # silently fall back to XLA's default preallocation fraction
        # (unreachable at production grid sizes; toy-scale guard only).
        mem = max(min(arena_mib / total_mib, SAFETY / wpg), 1e-4)
    else:
        mem = SAFETY / wpg
    diag = (f"adaptive packing: free~{free_mib}MiB/GPU, {basis}, "
            f"{len(tasks)} grids "
            f"-> {wpg} worker(s)/GPU ({mode}, per-worker mem_fraction={mem:.3f})")
    return wpg, mode, mem, diag


def _run_grids_on_slots(
    tasks: list, slot_devices: list[int], ctx, alloc_mode: str,
    per_worker_mem: float, jit_cache_dir, worker_log_file, disable_warmup: bool,
    defer_shutdown: bool = False,
) -> "tuple[list, threading.Thread | None]":
    """Run ``tasks`` across worker slots (one ProcessPool/slot, each pinned to
    its GPU). Tasks are pulled from ONE shared queue ordered by descending
    source count (LPT scheduling): the old per-slot round-robin pre-assignment
    let one straggler grid idle its sibling slots for ~30s/run, while with a
    shared queue the most expensive grids start first and the tail evens out.
    Per-grid math is independent + deterministic, so the result set is
    identical regardless of slot count/assignment -- this only changes
    scheduling.

    ``defer_shutdown=True`` moves worker teardown (each worker releasing a
    multi-GiB BFC arena under MPS; ~14s wall for 8 slots, measured) onto a
    background thread and returns it alongside the results. The caller MUST
    join that thread before spawning replacement workers (the OOM-retry pass
    needs the GPU memory back) but can otherwise let teardown overlap the
    merge + ZP stages. Reusable for the OOM-retry pass (1 slot/GPU)."""
    from concurrent.futures import FIRST_COMPLETED, Future
    from concurrent.futures import wait as futures_wait
    n_slots = len(slot_devices)
    executors: list[tuple[int, ProcessPoolExecutor]] = []
    ping_futs: list[tuple[int, Future]] = []
    for s in range(n_slots):
        d = slot_devices[s]
        ex = ProcessPoolExecutor(
            max_workers=1, mp_context=ctx, initializer=_worker_init,
            initargs=(d, per_worker_mem, jit_cache_dir, worker_log_file, alloc_mode),
        )
        if not disable_warmup:
            ping_futs.append((d, ex.submit(_worker_ping)))
        executors.append((s, ex))
    # Bring all workers up (create CUDA contexts) before sending multi-GB
    # payloads, so the first big submit doesn't stagger worker startup.
    if not disable_warmup:
        for d, fut in ping_futs:
            seen = int(fut.result())
            if seen != int(d):
                logger.warning("grid_gpu: worker warmup for GPU %d reported GPU %d", d, seen)
    # LPT order: descending source count. Actual cost tracks the (unknown
    # upfront) S3 iteration count, but starting big grids first still cuts
    # the straggler tail vs arbitrary order (simulated on measured per-grid
    # times: ~234s round-robin -> ~219s).
    queue = deque(sorted(
        tasks, key=lambda t: int(t.source_indices.shape[0]), reverse=True,
    ))
    pending: dict[Future, tuple[int, ProcessPoolExecutor]] = {}
    for s, ex in executors:
        if queue:
            pending[ex.submit(run_one_grid_managed, queue.popleft())] = (s, ex)
    results: list = []
    teardown_thread: threading.Thread | None = None
    completed_clean = False
    try:
        while pending:
            done, _not_done = futures_wait(set(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                s, ex = pending.pop(fut)
                res = fut.result()
                results.append(res)
                logger.info(
                    "grid_gpu: grid %d done in %.1fs (n_sources=%d, S3 iters=%d, "
                    "S1=%s S2=%s S3=%s%s)",
                    res.grid_id, res.sec_total, res.n_sources, res.n_outer_iters,
                    res.s1_converged, res.s2_converged, res.s3_converged,
                    f", error={res.error}" if res.error else "",
                )
                if queue:
                    pending[ex.submit(run_one_grid_managed, queue.popleft())] = (s, ex)
        completed_clean = True
    finally:
        def _teardown() -> None:
            for _s, ex in executors:
                ex.shutdown(wait=True)

        if defer_shutdown and completed_clean:
            # Only defer on a CLEAN pass: on an exception/KeyboardInterrupt
            # this tuple return never reaches the caller, so a deferred
            # thread would be orphaned and the caller's finally would stop
            # MPS while packed workers are still alive (leaking the daemon
            # if the quit times out). Synchronous teardown on error paths
            # keeps the pre-patch semantics.
            teardown_thread = threading.Thread(
                target=_teardown, name="tract7dt-slot-teardown",
            )
            teardown_thread.start()
        else:
            _teardown()
    return results, teardown_thread


def _mps_control(cmd: str, pipe_dir: str, timeout: float = 8.0) -> tuple[bool, str]:
    """Send one command to nvidia-cuda-mps-control on `pipe_dir`. (ok, stdout)."""
    import subprocess
    env = dict(os.environ)
    env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    try:
        p = subprocess.run(["nvidia-cuda-mps-control"], input=cmd + "\n",
                           capture_output=True, text=True, timeout=timeout, env=env)
        return (p.returncode == 0), (p.stdout or "")
    except Exception:  # noqa: BLE001
        return False, ""


# Track an MPS daemon WE started so it's cleaned up even on abnormal exit
# (atexit) or SIGTERM/pkill. The normal path stops it in the dispatch `finally`;
# these are backstops. SIGKILL (-9) is inherently uncleanable by any process.
_MPS_OWNED_PIPE: "str | None" = None
_MPS_CLEANUP_INSTALLED = False


def _mps_cleanup_owned() -> None:
    """Stop the MPS daemon we started, if any. Idempotent."""
    global _MPS_OWNED_PIPE
    pipe = _MPS_OWNED_PIPE
    if not pipe:
        return
    _MPS_OWNED_PIPE = None
    try:
        _mps_control("quit", pipe, timeout=10)
    except Exception:  # noqa: BLE001 - best-effort at shutdown
        pass


_SIGTERM_RAISE_INSTALLED = False


def _install_sigterm_raise() -> None:
    """Make SIGTERM raise SystemExit so `finally`/atexit cleanup runs.

    The handler only RAISES (async-signal-safe); the real cleanup (shm
    unlink, MPS daemon quit) runs in normal context via the dispatch
    `finally` and the atexit backstops. Shared by the MPS daemon and the
    grid-shm backstops. Idempotent; SIGKILL (-9) cannot be caught."""
    global _SIGTERM_RAISE_INSTALLED
    if _SIGTERM_RAISE_INSTALLED:
        return
    _SIGTERM_RAISE_INSTALLED = True
    try:
        _prev = signal.getsignal(signal.SIGTERM)

        def _sigterm_handler(signum, frame):
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                try:
                    _prev(signum, frame)
                except SystemExit:
                    pass
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        # Not in the main thread (signals unavailable) -> atexit still covers
        # normal/exception/sys.exit paths.
        pass


def _install_mps_cleanup() -> None:
    """Register atexit + SIGTERM backstops so a pkill/SIGTERM or abnormal exit
    doesn't leak a daemon we started."""
    global _MPS_CLEANUP_INSTALLED
    if _MPS_CLEANUP_INSTALLED:
        return
    _MPS_CLEANUP_INSTALLED = True
    atexit.register(_mps_cleanup_owned)
    _install_sigterm_raise()


def _dispatch_grids_packed(
    tasks: list, devices: list[int], gpu_cfg: dict, ctx, jit_cache_dir,
    worker_log_file, label: str,
) -> list:
    """Run S1-only grid tasks with the SAME measured packing the final grid
    pass uses (G1), instead of one exclusive worker per GPU.

    NOTE the parent is NOT JAX-free after this call: `_choose_workers_per_gpu`
    probes the S1 kernel in-process, so a CUDA context (~300 MiB/GPU measured)
    persists into the island stage and is subtracted from the shard workers'
    free-memory budget (<3% of the 11.83 GiB they see; adopted widths verified
    unchanged). The grid SOLVES themselves never run in the parent (no
    solo-inline branch, here or in the final pass since R17), which would hold
    multi-GiB arenas across the island stage.

    effarc2 K7 (2026-08-26). The island architecture's PASS 1 (per-grid S1
    seed solve) was dispatched with ``list(devices)`` -- one worker per GPU,
    ``platform`` allocator, no MPS, no OOM retry -- while PASS 2, over the
    SAME grids doing strictly MORE work per grid, packs up to 8 workers per
    GPU on the measured budget. On the crowded anchor that is 287 s (pass A)
    + 322 s (pass B) at 3 workers against 148/180 grids. It was also the
    registered "island pass-1 bypasses the safety net" gap (development
    ledger SS19-1): a pass-1 grid that OOM'd had no retry.

    Packing changes SCHEDULING, not the per-grid problem: each grid is an
    independent, deterministic solve merged by ``source_indices``, which is the
    argument the final pass already relies on. It is NOT a bit-identity claim:
    packed workers get a BFC arena instead of the on-demand allocator and share
    the card through MPS, and XLA autotuning can select differently under
    different memory/occupancy conditions -- the documented cold-run fp channel.

    The parent must stay JAX-FREE here (the island stage forks its own
    JAX-free shard workers right after), so a single slot still runs in a
    subprocess.
    """
    wpg, alloc_mode, per_worker_mem, wpg_diag = _choose_workers_per_gpu(
        tasks, devices, gpu_cfg)
    logger.info("grid_gpu[%s]: %s", label, wpg_diag)
    # Never more slots than grids: an empty slot still spawns a process, opens
    # a CUDA context and preallocates its arena (the main dispatch applies the
    # same cap).
    n_slots = max(1, min(len(devices) * wpg, len(tasks)))
    for t in tasks:
        t.n_grid_slots = int(n_slots)
    disable_warmup = os.environ.get(
        "TRACT7DT_BENCH_DISABLE_WORKER_WARMUP") == "1"
    use_mps = (str(os.environ.get("TRACT7DT_GPU_USE_MPS",
                                  str(gpu_cfg.get("use_mps", True))))
               .strip().lower() not in ("0", "false", "no", "off")) and wpg > 1
    mps_active, mps_started, mps_pipe = _maybe_start_mps(use_mps)
    slot_devices = [devices[s % len(devices)] for s in range(n_slots)]
    try:
        results, _ = _run_grids_on_slots(
            tasks, slot_devices, ctx, alloc_mode, per_worker_mem,
            jit_cache_dir, worker_log_file, disable_warmup,
        )
        oom = [r for r in results if _is_oom_error(r)]
        if oom:
            by_grid = {t.grid_id: t for t in tasks}
            retry_tasks = [by_grid[r.grid_id] for r in oom
                           if r.grid_id in by_grid]
            logger.warning(
                "grid_gpu[%s]: %d grid(s) OOM'd under %dw/GPU packing; "
                "retrying 1 worker/GPU (exclusive memory): grids=%s",
                label, len(retry_tasks), wpg,
                sorted(r.grid_id for r in oom))
            for t in retry_tasks:
                t.n_grid_slots = len(devices)
            retry_results, _ = _run_grids_on_slots(
                retry_tasks, list(devices), ctx, "platform",
                float(gpu_cfg.get("mem_fraction", 0.5)),
                jit_cache_dir, worker_log_file, disable_warmup,
            )
            rmap = {r.grid_id: r for r in retry_results}
            results = [rmap.get(r.grid_id, r) if _is_oom_error(r) else r
                       for r in results]
        return results
    finally:
        # NOTE: the published per-band /dev/shm segments are NOT cleaned here
        # -- pass 2 attaches to the same segments; the parent's own
        # `_cleanup_grid_shm` in run_grid_gpu_fit owns their lifetime.
        _stop_mps(mps_started, mps_pipe)


def _mps_daemon_env(pipe_dir: str) -> "dict | None":
    """Environment of this user's MPS control daemon serving ``pipe_dir``,
    read from /proc (None when no such process is found or readable)."""
    uid = os.getuid() if hasattr(os, "getuid") else 0
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                if b"nvidia-cuda-mps-control" not in fh.read():
                    continue
            with open(f"/proc/{pid}/environ", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        env = {}
        for kv in raw.split(b"\0"):
            k, sep, v = kv.partition(b"=")
            if sep:
                env[k.decode(errors="replace")] = v.decode(errors="replace")
        if env.get("CUDA_MPS_PIPE_DIRECTORY") == pipe_dir:
            return env
    return None


def _mps_daemon_sees_all_cards(pipe_dir: str) -> bool:
    env = _mps_daemon_env(pipe_dir)
    return env is not None and not env.get("CUDA_VISIBLE_DEVICES", "").strip()


def _maybe_start_mps(enabled: bool) -> tuple[bool, bool, str | None]:
    """Start (or reuse) an NVIDIA MPS daemon so co-located packed workers run
    CONCURRENT GPU kernels instead of time-slicing. Uses a PRIVATE per-user
    pipe dir, so it never touches a system/other-user MPS daemon (shared-box
    safe). Returns (active, started_by_us, pipe_dir). Graceful: any failure ->
    (False, False, None) and packing simply time-slices (no harm)."""
    if not enabled:
        return False, False, None
    import shutil, subprocess
    if shutil.which("nvidia-cuda-mps-control") is None:
        logger.info("grid_gpu: MPS on but nvidia-cuda-mps-control not found; "
                    "packed workers will time-slice the GPU.")
        return False, False, None
    uid = os.getuid() if hasattr(os, "getuid") else 0
    pipe_dir = f"/tmp/tract7dt-mps-{uid}"
    log_dir = f"/tmp/tract7dt-mps-log-{uid}"
    # Reuse a daemon already on OUR private pipe (e.g. leftover from a prior
    # run) only if it sees every card (R12). A daemon started under a
    # CUDA_VISIBLE_DEVICES subset numbers the cards differently from the
    # workers, which pin PHYSICAL indices -> CUDA_ERROR_NO_DEVICE -> failed
    # grids (UDS 2026-08-31). The per-user run lock guarantees no other run
    # of ours is using a leftover daemon, so replacing it is safe.
    if _mps_control("get_default_active_thread_percentage", pipe_dir, timeout=5)[0]:
        if _mps_daemon_sees_all_cards(pipe_dir):
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
            os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
            logger.info("grid_gpu: reusing existing MPS daemon at %s", pipe_dir)
            return True, False, pipe_dir
        logger.warning("grid_gpu: the MPS daemon at %s does not see every card "
                       "(started under CUDA_VISIBLE_DEVICES, or unreadable); "
                       "replacing it", pipe_dir)
        _mps_control("quit", pipe_dir, timeout=10)
        for _ in range(20):             # the daemon exits asynchronously
            if not _mps_control("get_default_active_thread_percentage",
                                pipe_dir, timeout=5)[0]:
                break
            time.sleep(0.5)
        else:
            logger.warning("grid_gpu: the old MPS daemon at %s did not quit; "
                           "packing will time-slice (no harm).", pipe_dir)
            return False, False, None
    try:
        os.makedirs(pipe_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        env["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        # The daemon must see every card: workers pin physical indices.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env=env,
                       capture_output=True, text=True, timeout=15)
        if not _mps_control("get_default_active_thread_percentage", pipe_dir, timeout=5)[0]:
            logger.warning("grid_gpu: MPS daemon did not come up at %s; "
                           "packing will time-slice (no harm).", pipe_dir)
            return False, False, None
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
        os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
        global _MPS_OWNED_PIPE
        _MPS_OWNED_PIPE = pipe_dir          # mark owned for atexit/SIGTERM backstops
        _install_mps_cleanup()
        logger.info("grid_gpu: started private MPS daemon at %s (concurrent "
                    "multi-process GPU kernels for packing)", pipe_dir)
        return True, True, pipe_dir
    except Exception as exc:  # noqa: BLE001
        logger.warning("grid_gpu: could not start MPS (%s); packing will "
                       "time-slice (no harm).", exc)
        return False, False, None


def _stop_mps(started_by_us: bool, pipe_dir: str | None) -> None:
    """Quit only an MPS daemon WE started (leave pre-existing daemons alone)."""
    global _MPS_OWNED_PIPE
    if not started_by_us or not pipe_dir:
        return
    ok = _mps_control("quit", pipe_dir, timeout=10)[0]
    _MPS_OWNED_PIPE = None  # cleared so the atexit/SIGTERM backstops don't re-quit
    logger.info("grid_gpu: stopped our MPS daemon at %s (%s)",
                pipe_dir, "ok" if ok else "quit failed")
    for k in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY"):
        os.environ.pop(k, None)


def _project_catalog_to_pixels(
    catalog: pd.DataFrame, image_dict: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Project (RA, DEC) -> (x_pix_white, y_pix_white) using the white
    image's WCS (0-based pixel origin).
    """
    col_map = case_insensitive_col_map(catalog.columns)
    ra_col = col_map.get("ra")
    dec_col = col_map.get("dec")
    if ra_col is None or dec_col is None:
        raise ValueError("input_catalog must have RA/DEC columns")
    ra = pd.to_numeric(catalog[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(catalog[dec_col], errors="coerce").to_numpy(dtype=float)
    wcs_white = next(iter(image_dict.values()))["wcs"]
    ok = np.isfinite(ra) & np.isfinite(dec)
    xpix = np.full(len(ra), np.nan, dtype=float)
    ypix = np.full(len(dec), np.nan, dtype=float)
    if np.any(ok):
        xw, yw = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
        xpix[ok] = xw
        ypix[ok] = yw
    return xpix.astype(np.float32), ypix.astype(np.float32)


def _build_per_band_arrays(
    band_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Extract (data, invvar, pix_good, sky_baseline) from one band's
    image_dict entry.

    The data has the SKY baked in (img_scaled is just scaled, NOT
    sky-subtracted). For grid_gpu we subtract a per-band sky baseline
    so the polynomial bg fits only SPATIAL DEVIATIONS from the sky,
    not the entire sky level. This is purely a numerical preconditioning
    step: source fluxes are INVARIANT to the choice of sky_baseline
    (the bg polynomial's constant term absorbs any difference).

    Sky baseline source order:
    1. ``hdr["SKYVAL"]`` * ``band_data["scale"]`` (microseconds).
       The 7DT pipeline writes "SKY MEDIAN VALUE" alongside SKYSIG.
       Even when SKYVAL differs from a freshly-computed median by 30-50%
       (e.g. computed on uncropped image vs our cropped region), source
       fluxes are unaffected because the bg const coef compensates.
    2. Fallback: 3-sigma-clipped median over good pixels (~3 s/band on
       a 53M-pixel image). Used only when SKYVAL is missing/non-finite.
    """
    data = np.asarray(band_data["img_scaled"], dtype=np.float32)
    # CRITICAL: use sigma_sky (sky-only noise), NOT sigma_total which
    # includes source Poisson noise. Using sigma_total downweights bright
    # source pixels and biases LM forced photometry toward LOWER fluxes
    # (~50% undershoot in our tests). Tractor / CPU per-patch path also
    # uses sigma_sky for forced phot (see tract7dt/fit/runner.py
    # use_sky_sigma=True default).
    sigma = np.asarray(band_data["sigma_sky_scaled"], dtype=np.float32)
    bad = np.asarray(band_data["bad"], dtype=bool)
    # astype(copy=False): the `where` results are already float32, so the
    # dtype casts are no-ops -- copy=True would transiently double the
    # ~264 MB/band allocations across the 16 concurrent build threads.
    invvar = np.where(sigma > 0, 1.0 / np.maximum(sigma * sigma, 1e-30), 0.0).astype(np.float32, copy=False)
    pix_good = (~bad) & np.isfinite(data) & np.isfinite(invvar) & (invvar > 0)
    invvar = np.where(pix_good, invvar, 0.0).astype(np.float32, copy=False)

    n_good = int(pix_good.sum())
    if n_good <= 100:
        sky_baseline = 0.0
        logger.warning("  band data has only %d good pixels; sky_baseline=0", n_good)
        data_sub = (data - np.float32(sky_baseline)).astype(np.float32, copy=False)
        return data_sub, invvar, pix_good, sky_baseline

    # FAST PATH: read sky baseline from the FITS header (saves 67% of
    # this stage's wall time vs computing 3 medians on 53M pixels).
    hdr = band_data.get("hdr")
    scale = band_data.get("scale", 1.0)
    sky_val_raw = None
    if hdr is not None:
        sky_val_raw = hdr.get("SKYVAL")
        if sky_val_raw is None or not np.isfinite(float(sky_val_raw)):
            sky_val_raw = None
    if sky_val_raw is not None:
        sky_baseline = float(sky_val_raw) * float(scale)
        logger.info(
            "  band data stats: n_good=%d, sky_baseline=%.4g (header SKYVAL=%.4g, scale=%.4g)",
            n_good, sky_baseline, float(sky_val_raw), float(scale),
        )
    else:
        # SLOW PATH (fallback): 3-sigma-clipped median over good pixels.
        gd = data[pix_good]
        med = float(np.median(gd))
        mad = float(np.median(np.abs(gd - med))) * 1.4826
        clip_lo, clip_hi = med - 3.0 * mad, med + 3.0 * mad
        gd_clip = gd[(gd >= clip_lo) & (gd <= clip_hi)]
        if gd_clip.size > 100:
            med = float(np.median(gd_clip))
        sky_baseline = med
        logger.info(
            "  band data stats: n_good=%d, min=%.4g, max=%.4g, "
            "median=%.4g, mad=%.4g, baseline=%.4g (computed -- header SKYVAL absent)",
            n_good, float(gd.min()), float(gd.max()), med,
            float(mad), sky_baseline,
        )
    data_sub = (data - np.float32(sky_baseline)).astype(np.float32, copy=False)
    return data_sub, invvar, pix_good, sky_baseline


def _compute_slow_converge_flag(
    opt_niters: np.ndarray,
    has_fit: np.ndarray,
    max_steps_main,
    max_steps,
) -> np.ndarray:
    """flag_slow_converge: the source's island did NOT converge within the
    solver's fast MAIN-PASS budget (island solver only).

    ``flag = opt_niters > max_steps_main`` -- the solver's OWN boundary between a
    fast converger and a re-queued straggler (the #25 difficulty re-queue: islands
    still moving after ``max_steps_main`` steps are re-queued into the hard pass;
    see island_solver). CONFIG-derived from the solver's existing convergence
    budget, robust to the field's convergence distribution, and NOT calibrated to
    injection truth -- this replaces the truth-tuned ``slow_converge_niters: 12``.
    When the re-queue is disabled (``max_steps_main`` is None/0 or >= ``max_steps``)
    there is no fast/slow split, so it falls back to ``opt_niters >= max_steps``
    (the hard iteration cap = genuine non-convergence). Unfitted rows and rows on
    a field with no usable budget are never flagged.
    """
    opt_niters = np.asarray(opt_niters, dtype=float)
    has_fit = np.asarray(has_fit, dtype=bool)
    ms = int(max_steps) if max_steps is not None else 0
    msm = int(max_steps_main) if max_steps_main is not None else 0
    if msm > 0 and (ms <= 0 or msm < ms):
        thr = float(msm)          # > main-pass budget = a re-queued straggler
    elif ms > 0:
        thr = float(ms) - 1.0     # re-queue off: fall back to opt_niters >= max_steps
    else:
        return np.zeros(opt_niters.shape, dtype=bool)
    return (opt_niters > thr) & has_fit


def blank_response_bhat_table(results, bandnames_with_psf) -> pd.DataFrame:
    """REGISTER R09: the blank-response offset each grid subtracted from its
    STAR fluxes, one row per (grid, band).

    ``b_hat`` is in the flux units of ``FLUX_{band}_fit`` (image scaled to
    ``image_scaling.zp_ref``). A STAR row's uncorrected flux is
    ``FLUX_{band}_fit + b_hat`` of its ``grid_id_fit``. Failed grids and runs
    with the correction off contribute no rows.
    """
    rows = []
    for res in results:
        if getattr(res, "error", None):
            continue
        bh = getattr(res, "blank_response_bhat", None)
        if bh is None:
            continue
        bh = np.asarray(bh, dtype=np.float64)
        for bi, bn in enumerate(bandnames_with_psf):
            if bi >= bh.shape[0]:
                break
            rows.append({"grid_id": int(res.grid_id), "band": bn,
                         "b_hat": float(bh[bi])})
    return pd.DataFrame(rows, columns=["grid_id", "band", "b_hat"])


def run_grid_gpu_fit(cfg: dict[str, Any], state: dict[str, Any]) -> Path:
    """Run the grid-based GPU joint fit on the full image.

    See :file:`docs/grid-gpu-roadmap.md` for architecture.
    """
    t_total = time.perf_counter()
    compute_cfg = cfg.get("compute", {})
    gpu_cfg = compute_cfg.get("gpu", {}) or {}
    grid_layout = gpu_cfg.get("grid", {}) or {}
    psf_cfg_block = gpu_cfg.get("psf", {}) or {}
    bg_block = gpu_cfg.get("background", {}) or {}
    solver_block = gpu_cfg.get("solver", {}) or {}
    positions_block = gpu_cfg.get("positions", {}) or {}

    n_rows = int(grid_layout.get("n_rows", 2))
    n_cols = int(grid_layout.get("n_cols", 2))
    halo_pix = int(grid_layout.get("halo_pix", 30))
    max_sources_per_grid_raw = grid_layout.get("max_sources_per_grid", 1500)
    psf_stamp_size = int(psf_cfg_block.get("stamp_size", 55))
    bg_poly_order = int(bg_block.get("poly_order", 1))
    # NOTE: compute.gpu.positions.default_pos_err_pix is DEPRECATED and
    # NO LONGER READ (2026-07-08). An ABSENT POS_ERR column now means NO
    # position prior (data-driven fit), matching the CPU path -- a seed
    # is a starting point, not a belief. See the POS_ERR block below.
    pos_max_shift_pix = float(positions_block.get("pos_max_shift_pix", 5.0))
    mem_fraction = float(gpu_cfg.get("mem_fraction", 0.5))
    jit_cache_dir = gpu_cfg.get("jit_cache_dir", "auto")
    single_stage = bool(solver_block.get("single_stage", False))
    n_opt_iters = int(solver_block.get("n_opt_iters", 20))
    pos_stop_unconverged_frac = float(solver_block.get("pos_stop_unconverged_frac", 0.0))
    chi2_ftol = float(solver_block.get("chi2_ftol", 0.1))
    pos_step_max_pix = float(solver_block.get("pos_step_max_pix", 1.0))
    dlnp_stop_pix = float(solver_block.get("dlnp_stop_pix", 0.1))
    # NB: the "per_source" fallback is dead defense-in-depth -- via load_config
    # this key is always present (DEFAULT_CONFIG is the single source of truth
    # for the shipped default, currently "island"). Do not read this literal as
    # the default.
    position_solver = str(solver_block.get("position_solver", "per_source"))
    joint_cg_max_iters = int(solver_block.get("joint_cg_max_iters", 20))
    joint_cg_tol = float(solver_block.get("joint_cg_tol", 1e-6))
    joint_use_preconditioner = bool(solver_block.get("joint_use_preconditioner", True))
    if position_solver not in ("auto", "per_source", "joint_lsqr",
                               "joint_varpro", "island"):
        raise ValueError(
            f"compute.gpu.solver.position_solver must be 'auto', 'per_source', "
            f"'joint_lsqr', 'joint_varpro', or 'island'; got {position_solver!r}"
        )
    # ---- Island-architecture mode (OPT-IN; replaces S2/S3) ----------
    # Both per-grid passes (S1 seed solve + S1' final solve) are
    # S1-only: single_stage with ZERO joint iterations. The nonlinear
    # (position/shape/Sersic-n) fitting happens in the field-level
    # island stage between them (see island_stage.py).
    island_mode = position_solver == "island"
    island_cfg = solver_block.get("island", {}) or {}
    # 2026-07-22: thread the field-population declaration (the SAME
    # bright_mask.unmatched_population deep-field declaration that gates
    # subtract_unmatched) into the island stage so crossband in-run S can
    # warn on galaxy-dominated fields (P1b: in-run S there learns the
    # science population's own color; measured-harmful, ban documented).
    island_cfg = dict(island_cfg)
    island_cfg["_field_population"] = str(
        ((cfg.get("bright_mask", {}) or {})
         .get("unmatched_population", "stars"))).lower()
    if island_mode:
        single_stage = True
        n_opt_iters = 0
        logger.info(
            "grid_gpu: position_solver=island -- per-grid passes run "
            "S1-only (single_stage, 0 joint iters); S2/S3 are replaced "
            "by the field-level island stage (kappa=%s, max_steps=%s, "
            "max_steps_main=%s, joint_nsrc_cap=%s)",
            island_cfg.get("kappa", 0.5), island_cfg.get("max_steps", 30),
            island_cfg.get("max_steps_main", 8),
            island_cfg.get("joint_nsrc_cap", 32),
        )

    outputs = cfg["outputs"]
    out_csv = Path(outputs["final_catalog"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    image_dict = state["image_dict"]
    bandnames = sorted(image_dict.keys())
    catalog = state["input_catalog"]

    # ---- Exp C v2: phantom subtract-only sources ----
    # (bright_mask.subtract_unmatched_mode: "s1") Routed catalog-unmatched
    # bright detections become flux-only STAR columns of the S1/S1' JOINT
    # solves: their fluxes are solved SELF-CONSISTENTLY with the background
    # (G1 guard -- fixed-flux pre-subtraction is the measured-negative v1
    # form: under-subtraction residuals inflate the L2 background). They
    # are forced-singleton islands (never joint-fit with real sources),
    # scene-subtracted like any non-member STAR, and DROPPED from the
    # output catalog below (user-catalog purity; ID prefix sub_).
    phantom_mask = None
    _bm_sub = state.get("bright_mask_sub")
    _bm_mode = str((cfg.get("bright_mask", {}) or {}).get(
        "subtract_unmatched_mode", "sequential"))
    if _bm_sub is not None and len(_bm_sub) and _bm_mode == "s1":
        _wcs_ph = next(iter(image_dict.values()))["wcs"]
        _pra, _pdec = _wcs_ph.all_pix2world(
            np.asarray(_bm_sub[:, 0], dtype=float),
            np.asarray(_bm_sub[:, 1], dtype=float), 0)
        catalog = catalog.copy()
        if "_is_phantom_sub" not in catalog.columns:
            catalog["_is_phantom_sub"] = False
        _phmap = case_insensitive_col_map(catalog.columns)
        _ph_rows = pd.DataFrame({
            _phmap.get("id", "ID"): [f"sub_{i:05d}"
                                     for i in range(len(_bm_sub))],
            _phmap.get("ra", "RA"): _pra,
            _phmap.get("dec", "DEC"): _pdec,
            _phmap.get("type", "TYPE"): "STAR",
            "_is_phantom_sub": True,
        })
        for _c in ("excluded_crop", "excluded_saturation", "excluded_any",
                   "affected_by_bright_mask", "_is_injected_gaia"):
            if _c in catalog.columns:
                _ph_rows[_c] = False
        catalog = pd.concat([catalog, _ph_rows], ignore_index=True)
        phantom_mask = catalog["_is_phantom_sub"].fillna(
            False).to_numpy(dtype=bool)
        logger.info(
            "grid_gpu: Exp C v2: appended %d phantom subtract-only STAR "
            "column(s) (bright-mask-routed detections; joint S1 solve, "
            "forced-singleton islands, dropped from the output catalog)",
            len(_bm_sub))

    n_total = len(catalog)
    image_h, image_w = image_dict[bandnames[0]]["img_scaled"].shape
    logger.info(
        "grid_gpu: %d bands, image %dx%d, %d catalog sources, grid %dx%d, halo %d px",
        len(bandnames), image_h, image_w, n_total, n_rows, n_cols, halo_pix,
    )

    # ---- Project catalog to pixel coords ----
    t0 = time.perf_counter()
    xpix, ypix = _project_catalog_to_pixels(catalog, image_dict)
    catalog = catalog.copy()
    catalog["x_pix_white"] = xpix
    catalog["y_pix_white"] = ypix

    # ---- disclosure: distance to the nearest subtract-only source --------
    # `_bm_sub` is the routed subtract-only set (bright-mask unmatched
    # detections; until M2 unit 0.1 also the residual-phantom specks of the
    # two-pass form, where the measurement below was made). All of
    # them are modelled as flux-only STAR columns and scene-subtracted at a
    # jointly-solved flux, then dropped from the catalog. Which specks exist is
    # decided by SEP on a residual image that is not bit-reproducible, so a
    # source close to a speck inherits that instability -- and nothing in the
    # output said which sources those are. Measured on the crowded anchor
    # between two same-code runs: near a FLIPPED speck the pull
    # |dF| / (sqrt(2) sigma_F) is 2.8-3.2 at 0-5 px with 93% of sources over
    # pull 1, and the phantom correction itself is -14.5% at 3-4 px falling to
    # +1.5% beyond 15 px; ~138 RELIABLE sources sit within one PSF FWHM of a
    # speck (development ledger SS24). That is an error-model coverage gap for
    # a named subpopulation; this column is what lets anyone find them.
    #
    # Pure geometry: a distance, no radius, no threshold, no constant, and it
    # feeds no decision. Units are working white-frame pixels -- the same
    # frame as `x_pix_white`, which is what `_bm_sub[:, :2]` is stored in.
    # NaN on the phantom rows themselves (they are the subtracted set, not
    # neighbours of it) and when nothing is routed. NOTE: this adds a column to output_catalog.csv on EVERY run, so a
    # gate that byte-compares the catalog must be rebaselined once.
    if _bm_sub is not None and len(_bm_sub):
        from scipy.spatial import cKDTree as _cKDT
        _ph_tree = _cKDT(np.asarray(_bm_sub[:, :2], dtype=float))
        _sxy = np.column_stack([np.asarray(xpix, dtype=float),
                                np.asarray(ypix, dtype=float)])
        _fin = np.isfinite(_sxy).all(axis=1)
        _d_ph = np.full(len(catalog), np.nan)
        if _fin.any():
            _d_ph[_fin], _ = _ph_tree.query(_sxy[_fin])
        catalog["d_phantom_pix"] = (np.where(phantom_mask, np.nan, _d_ph)
                                    if phantom_mask is not None else _d_ph)
    else:
        catalog["d_phantom_pix"] = np.nan

    logger.info("grid_gpu: projected catalog (%d sources) in %.2fs",
                n_total, time.perf_counter() - t0)

    # ---- POS_ERR position prior (opt-in; a seed is NOT a belief) ----
    # 1e6 is the "no Gaussian prior" sentinel (legacy solver_grid uses
    # 1/max(pos_err,1e-6) ~ 0). Both a present-but-blank/invalid cell AND
    # an ABSENT column now map to it -> NO seed-pull position prior,
    # matching the CPU path (the old default_pos_err_pix=1px auto-pull
    # was removed 2026-07-08). A finite, positive POS_ERR is the
    # sanctioned opt-in: "this catalog position IS my belief, this wide".
    if "POS_ERR" in catalog.columns:
        pos_err_raw = pd.to_numeric(catalog["POS_ERR"], errors="coerce").to_numpy(dtype=np.float32)
        pos_err = np.where(
            np.isfinite(pos_err_raw) & (pos_err_raw > 0),
            pos_err_raw,
            1.0e6,
        ).astype(np.float32)
    else:
        pos_err = np.full(n_total, 1.0e6, dtype=np.float32)

    # ---- Source models: STAR/PointSource + EXP galaxies ----
    col_map = case_insensitive_col_map(catalog.columns)
    type_col_name = col_map.get("type")
    if type_col_name:
        supported_type = catalog[type_col_name].apply(is_supported_source_type).to_numpy(dtype=bool)
    else:
        supported_type = np.ones(n_total, dtype=bool)
    source_type, logre_init, ee1_init, ee2_init, sersic_n_init = catalog_source_params(
        catalog,
        re_fallback_pix=float(cfg.get("patch_run", {}).get("re_fallback_pix", 3.0)),
    )
    n_star = int(np.sum((source_type == SOURCE_STAR) & supported_type))
    n_exp = int(np.sum((source_type == SOURCE_EXP) & supported_type))
    n_dev = int(np.sum((source_type == SOURCE_DEV) & supported_type))
    n_ser = int(np.sum((source_type == SOURCE_SERSIC) & supported_type))
    n_skip_type = n_total - int(np.sum(supported_type))
    logger.info(
        "grid_gpu: source models: STAR/point=%d, EXP=%d, DEV=%d, SERSIC=%d, "
        "skipped unsupported=%d",
        n_star, n_exp, n_dev, n_ser, n_skip_type,
    )

    # Renderer choice: "auto" keeps the historical direct path for pure
    # point-source catalogs (bit-identical) and uses the analytic-flux FFT
    # renderer whenever extended sources are present. DEV/SERSIC require fft.
    render_backend_cfg = str(
        cfg.get("compute", {}).get("gpu", {}).get("galaxy", {})
        .get("render_backend", "auto")
    ).lower()
    any_extended = bool(np.any((source_type != SOURCE_STAR) & supported_type))
    if render_backend_cfg == "auto":
        render_backend = "fft" if any_extended else "direct"
    else:
        render_backend = render_backend_cfg
    if render_backend == "direct" and (n_dev + n_ser) > 0:
        raise ValueError(
            "compute.gpu.galaxy.render_backend='direct' cannot render DEV/"
            "SERSIC sources; use 'fft' or 'auto'."
        )
    logger.info("grid_gpu: galaxy render_backend=%s (config: %s)",
                render_backend, render_backend_cfg)

    # ---- Phase-B stamp-tier promotion for large galaxies ----------------
    # A galaxy needs its render box to hold ~nre effective radii (Tractor
    # patch convention: nre=4 for disky, 8 for cuspy profiles); beyond the
    # box the wings are truncated and, worse, absorbed by the background
    # model (measured: -10..-14% flux at Re=16-40 px with the 55px stamp).
    # Per-source requirement -> per-grid tier (a grid renders every source
    # at its tier; implemented downstream by zero-padding the PSF stamp).
    galaxy_cfg = (gpu_cfg.get("galaxy", {}) or {})
    # Stage-2 free-n knobs (varpro): fit TYPE=SERSIC Sersic index for rows
    # whose S1 flux S/N clears the gate; the rest stay frozen at the seed.
    fit_sersic_n = bool(galaxy_cfg.get("fit_sersic_n", True))
    n_fit_min_snr = float(galaxy_cfg.get("n_fit_min_snr", 30.0))
    # Wing peeling for large galaxies: subtract per-band wing images
    # (tier render minus native render) of big sources from the data;
    # replaces the OOM-prone whole-grid stamp-tier promotion.
    wing_peeling = bool(galaxy_cfg.get("wing_peeling", False))
    wing_peel_tier = int(galaxy_cfg.get("wing_peel_tier", 221))
    wing_peel_max_sources = int(galaxy_cfg.get("wing_peel_max_sources", 64))
    stamp_tiers = sorted(
        {int(t) for t in galaxy_cfg.get("stamp_tiers", [55, 111, 221])}
        | {int(psf_stamp_size)}
    )
    nre_arr = np.where(sersic_n_init < 1.5, 4.0,
                       np.where(sersic_n_init < 3.0, 6.0, 8.0))
    req_half = np.where(
        (source_type != SOURCE_STAR) & supported_type,
        nre_arr * np.exp(logre_init.astype(float)), 0.0)
    src_tier = np.full(n_total, stamp_tiers[-1], dtype=np.int32)
    for t in reversed(stamp_tiers):
        src_tier = np.where(req_half <= t // 2, t, src_tier).astype(np.int32)
    n_promoted = int(np.sum(src_tier > stamp_tiers[0]))
    if n_promoted:
        logger.info(
            "grid_gpu: stamp tiers %s: %d source(s) promoted above %d px "
            "(max required half-width %.0f px)",
            stamp_tiers, n_promoted, stamp_tiers[0], float(req_half.max()))
        # Grid halo must cover the LARGEST promoted stamp's half-width so
        # cross-grid sources keep a full stamp inside grid+halo.
        halo_pix = max(halo_pix, int(max(src_tier)) // 2 + 3)

    if "excluded_any" in catalog.columns:
        excluded_any = catalog["excluded_any"].fillna(False).astype(bool).to_numpy()
    else:
        excluded_any = np.zeros(n_total, dtype=bool)
    if "affected_by_bright_mask" in catalog.columns:
        affected_by_bright_mask = (
            catalog["affected_by_bright_mask"].fillna(False).astype(bool).to_numpy()
        )
    else:
        affected_by_bright_mask = np.zeros(n_total, dtype=bool)

    # Sources with non-finite pixel coords, hard exclusions, or bright-mask
    # overlap are skipped entirely.  Bright-mask overlap means most of the
    # local pixels have zero weight; fitting those rows adds poorly
    # constrained columns that can wander and slow or destabilize the grid fit.
    has_pix = np.isfinite(xpix) & np.isfinite(ypix)
    keep = supported_type & has_pix & (~excluded_any) & (~affected_by_bright_mask)
    keep_idx = np.where(keep)[0]
    logger.info(
        "grid_gpu: fitting %d/%d sources (skip no_pix=%d, excluded=%d, bright_mask=%d)",
        int(keep.sum()),
        n_total,
        int(np.sum(supported_type & ~has_pix)),
        int(np.sum(supported_type & has_pix & excluded_any)),
        int(np.sum(supported_type & has_pix & ~excluded_any & affected_by_bright_mask)),
    )

    # ---- Load per-band PSF model ----
    # PSFEx-style polynomial PSF built upstream by
    # build_psf_polynomial_from_stars_stage (psf.mode from_stars, the only
    # mode since card 018). A band without one gets NaN fluxes.
    t0 = time.perf_counter()
    psfs: dict[str, Any] = {}
    psf_dir_str = state.get("psf_polynomial_dir")
    if psf_dir_str and Path(psf_dir_str).is_dir():
        from .psf_pipeline_stage import load_psf_polynomial_for_band
        for bn in bandnames:
            try:
                psfs[bn] = load_psf_polynomial_for_band(Path(psf_dir_str), bn)
            except FileNotFoundError:
                logger.warning(
                    "grid_gpu: PSF-from-stars missing for band %s", bn,
                )
        logger.info(
            "grid_gpu: loaded PSFEx-style polynomial PSFs for %d / %d "
            "bands from %s in %.2fs",
            len(psfs), len(bandnames), psf_dir_str,
            time.perf_counter() - t0,
        )
    logger.info("grid_gpu: PSF model (mode=%s) ready for %d / %d bands "
                "in %.2fs", "from_stars", len(psfs), len(bandnames),
                time.perf_counter() - t0)
    if len(psfs) < len(bandnames):
        still_missing = sorted(set(bandnames) - set(psfs.keys()))
        logger.warning("grid_gpu: %d bands have no polynomial PSF (skipped); "
                       "fluxes will be NaN: %s",
                       len(still_missing), still_missing)

    # CONSISTENT bg/PSF normalization across grids: derived from the
    # image shape (independent of PSF type). We use this for the bg
    # polynomial basis evaluation (PsfPolynomial happens to use the
    # same convention; PsfPerCellLookup doesn't carry these fields).
    bg_x_origin = 0.5 * (image_w - 1)
    bg_x_scale = max(0.5 * image_w, 1.0)
    bg_y_origin = 0.5 * (image_h - 1)
    bg_y_scale = max(0.5 * image_h, 1.0)

    # ---- Divide image into grids + assign sources ----
    max_sources_per_grid: int | None
    if max_sources_per_grid_raw is None:
        max_sources_per_grid = None
    elif isinstance(max_sources_per_grid_raw, str) and max_sources_per_grid_raw.strip().lower() in ("", "none", "null", "off", "false"):
        max_sources_per_grid = None
    else:
        max_sources_per_grid = int(max_sources_per_grid_raw)
        if max_sources_per_grid <= 0:
            max_sources_per_grid = None
    if max_sources_per_grid is not None and keep_idx.size > (n_rows * n_cols * max_sources_per_grid):
        min_cells = int(np.ceil(float(keep_idx.size) / float(max_sources_per_grid)))
        best: tuple[int, float, int, int] | None = None
        max_dim = max(n_rows, n_cols, int(np.ceil(np.sqrt(min_cells))) * 4)
        for rr in range(n_rows, max_dim + 1):
            cc_min = max(n_cols, int(np.ceil(min_cells / rr)))
            cc_max = max(cc_min, min(max_dim, cc_min + 3))
            for cc in range(cc_min, cc_max + 1):
                cells = rr * cc
                if cells < min_cells:
                    continue
                cell_aspect = (image_w / float(cc)) / max(image_h / float(rr), 1.0)
                aspect_err = abs(np.log(max(cell_aspect, 1e-6)))
                score = (cells, aspect_err, rr, cc)
                if best is None or score < best:
                    best = score
        if best is not None:
            old_rows, old_cols = n_rows, n_cols
            n_rows, n_cols = int(best[2]), int(best[3])
            logger.info(
                "grid_gpu: auto-refined grid %dx%d -> %dx%d for %d fitted sources "
                "(target <=%d owned sources/grid; set compute.gpu.grid.max_sources_per_grid=null to disable)",
                old_rows, old_cols, n_rows, n_cols, int(keep_idx.size), max_sources_per_grid,
            )

    grids = divide_image_into_grids(
        image_h=image_h, image_w=image_w,
        n_rows=n_rows, n_cols=n_cols, halo=halo_pix,
    )
    # Assign only the kept (PointSource + valid pixel coord) sources.
    assign_sources_to_grids(grids, xpix[keep_idx], ypix[keep_idx])
    for g in grids:
        # Map back to global catalog indices.
        g.source_indices = keep_idx[g.source_indices].astype(np.int64)
        g.halo_source_indices = keep_idx[g.halo_source_indices].astype(np.int64)
    owned_counts = [len(g.source_indices) for g in grids]
    fit_counts = [len(g.source_indices) + len(g.halo_source_indices) for g in grids]
    logger.info(
        "grid_gpu: assigned sources to %d grids: owned=%s, fit-with-halo=%s",
        len(grids), owned_counts, fit_counts,
    )

    # ---- Build per-band sliced arrays for each grid ----
    # We compute the per-band (data, invvar, pix_good) per band ONCE
    # at full image scale, then slice per grid. ALSO compute the
    # per-band sky baseline (median of good pixels) and subtract it
    # from data, so the polynomial bg fits only SPATIAL DEVIATIONS
    # from sky, not the whole sky level.
    t0 = time.perf_counter()
    per_band_full: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sky_baseline_per_band: dict[str, float] = {}
    # This stage is pure array preconditioning and independent per band.
    # Use a bounded thread pool: numpy releases the GIL for the large array
    # kernels, while the cap avoids excessive temporary memory pressure. Cap
    # raised 6->16 (memory-bandwidth bound past ~16; well within the core budget).
    band_array_workers = min(len(bandnames), max(1, os.cpu_count() or 1), 16)
    if band_array_workers <= 1:
        for bn in bandnames:
            data, invvar, pixg, sky_b = _build_per_band_arrays(image_dict[bn])
            per_band_full[bn] = (data, invvar, pixg)
            sky_baseline_per_band[bn] = sky_b
    else:
        logger.info(
            "grid_gpu: building per-band arrays with %d worker thread(s)",
            band_array_workers,
        )
        with ThreadPoolExecutor(max_workers=band_array_workers) as ex:
            fut_to_band = {
                ex.submit(_build_per_band_arrays, image_dict[bn]): bn
                for bn in bandnames
            }
            for fut in as_completed(fut_to_band):
                bn = fut_to_band[fut]
                data, invvar, pixg, sky_b = fut.result()
                per_band_full[bn] = (data, invvar, pixg)
                sky_baseline_per_band[bn] = sky_b
    logger.info("grid_gpu: built per-band arrays in %.2fs (sky baselines: "
                "%s)", time.perf_counter() - t0,
                ", ".join(f"{bn}={sky_baseline_per_band[bn]:.2f}"
                          for bn in bandnames[:5]) + (
                    "..." if len(bandnames) > 5 else ""
                ))

    # ---- Construct PerGridTask per grid ----
    bandnames_with_psf = [bn for bn in bandnames if bn in psfs]
    psfs_per_band_objs = [psfs[bn] for bn in bandnames_with_psf]
    psf_shape = (psfs[bandnames_with_psf[0]].H_psf, psfs[bandnames_with_psf[0]].W_psf)

    devices_cfg = gpu_cfg.get("devices", "auto")
    n_tasks_needed = sum(1 for g in grids if g.source_indices.size > 0)
    n_workers = min(n_tasks_needed, 16)
    devices = _resolve_gpu_devices(devices_cfg, n_workers)
    n_workers = len(devices)

    # Sweep leftovers from previous CRASHED runs before allocating: a kernel
    # OOM kill (SIGKILL) bypasses every in-process cleanup below, and leaked
    # segments are tmpfs-backed RAM that starves this run toward the same
    # fate (observed self-amplifying 62 GB leak on a shared box, 2026-06).
    sweep_stale_segments(logger)

    shm_handles: list[Any] = []
    shared_full_specs: dict[str, tuple[tuple[str, tuple[int, int], str], ...]] = {}
    use_shared_grid_payloads = False
    if (
        n_workers > 1
        and os.environ.get("TRACT7DT_BENCH_DISABLE_GRID_SHM") != "1"
    ):
        try:
            from multiprocessing import shared_memory
            _pid = os.getpid()

            def _publish_band_shm(bn: str) -> int:
                # Create + fill this band's 3 segments, then drop the
                # anon-RAM originals (the shm copy is the canonical data
                # now) so peak memory stays flat while tmpfs grows,
                # instead of holding both full sets (~2x13 GB at 22
                # bands) until dispatch.
                specs = []
                nbytes = 0
                for tag, arr in zip(("data", "invvar", "pixg"), per_band_full[bn]):
                    arr_c = np.ascontiguousarray(arr)
                    shm = shared_memory.SharedMemory(
                        create=True,
                        size=arr_c.nbytes,
                        name=f"tract7dt_grid_{bn}_{tag}_{_pid}",
                    )
                    # Publish BEFORE the fill so the except handler below
                    # sees (and releases) every segment created so far.
                    shm_handles.append(shm)
                    np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)[:] = arr_c
                    nbytes += int(arr_c.nbytes)
                    specs.append((shm.name, tuple(arr_c.shape), str(arr_c.dtype)))
                shared_full_specs[bn] = tuple(specs)
                per_band_full.pop(bn, None)
                return nbytes

            # The fill is a ~9 GiB GIL-releasing memcpy: copy bands
            # concurrently across threads (same pattern as the PSF stage;
            # measured ~8x on this host class, 6.5s -> ~1.2s).
            copy_workers = min(
                len(bandnames_with_psf), max(1, os.cpu_count() or 1), 8,
            )
            with ThreadPoolExecutor(max_workers=copy_workers) as _cp:
                total_shm_bytes = sum(_cp.map(_publish_band_shm, bandnames_with_psf))
            use_shared_grid_payloads = True
            # Bands without a PSF are never consumed on this path either.
            per_band_full.clear()
            logger.info(
                "grid_gpu: shared-memory grid payloads enabled (%.2f GiB across %d segments)",
                total_shm_bytes / (1024 ** 3), len(shm_handles),
            )
        except Exception as exc:
            logger.warning(
                "grid_gpu: shared-memory grid payload setup failed (%s); "
                "falling back to pickled grid slices",
                exc,
            )
            for shm in shm_handles:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
            shm_handles = []
            shared_full_specs = {}
            use_shared_grid_payloads = False
            # Rebuild any band arrays already dropped after their shm copy
            # (the pickled-slices fallback reads per_band_full directly).
            # Deterministic recompute from the unchanged image_dict; keep
            # the sky baselines already recorded above.
            for bn in bandnames_with_psf:
                if bn not in per_band_full:
                    data, invvar, pixg, _sky = _build_per_band_arrays(image_dict[bn])
                    per_band_full[bn] = (data, invvar, pixg)

    def _cleanup_grid_shm() -> None:
        for shm in shm_handles:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(
                    "grid_gpu: failed to clean shared memory %s: %s",
                    getattr(shm, "name", "<unknown>"), exc,
                )
        shm_handles.clear()  # second call (finally + atexit) is a no-op
        atexit.unregister(_cleanup_grid_shm)

    if shm_handles:
        # Backstops for exits that skip the dispatch `finally`: an exception
        # raised between here and the try below, sys.exit, and pkill/SIGTERM
        # (the handler raises SystemExit so `finally` + atexit both run).
        # SIGKILL (OOM killer) is uncatchable -- that hole is covered by
        # sweep_stale_segments() at the start of the NEXT run.
        atexit.register(_cleanup_grid_shm)
        _install_sigterm_raise()

    # Drop image_dict members that are dead from here on: sigma_* were
    # consumed by _build_per_band_arrays above (including the fallback
    # rebuild), satur_mask by the pre-fit stages (PSF build, Gaia
    # augmentation in zp.py). Together ~600 MB/band (~13 GB at 22 bands) of
    # anon RAM that would otherwise stay pinned through the GPU dispatch.
    # KEEP img_scaled/bad and wcs/hdr/scale. compute_zp, the only post-fit stage, does not read
    # image_dict.
    for _bd in image_dict.values():
        _bd.pop("sigma_total_scaled", None)
        _bd.pop("sigma_sky_scaled", None)
        _bd.pop("satur_mask", None)

    def _slice_spec(base: tuple[str, tuple[int, int], str], g: GridDef) -> SharedArraySlice:
        name, shape, dtype = base
        return SharedArraySlice(
            shm_name=name, shape=shape, dtype=dtype,
            y0=g.y0_pad, y1=g.y1_pad, x0=g.x0_pad, x1=g.x1_pad,
        )

    tasks: list[PerGridTask] = []
    auto_solver_counts = {"joint_varpro": 0, "per_source": 0}
    for g in grids:
        if g.source_indices.size == 0:
            logger.info("grid_gpu: grid %d (row=%d col=%d) has 0 sources; "
                        "skipping", g.grid_id, g.grid_row, g.grid_col)
            continue
        # Slice per-band arrays.
        data_list: list[np.ndarray] = []
        invvar_list: list[np.ndarray] = []
        pixg_list: list[np.ndarray] = []
        shared_data_list: list[SharedArraySlice] = []
        shared_invvar_list: list[SharedArraySlice] = []
        shared_pixg_list: list[SharedArraySlice] = []
        if use_shared_grid_payloads:
            for bn in bandnames_with_psf:
                data_spec, invvar_spec, pixg_spec = shared_full_specs[bn]
                shared_data_list.append(_slice_spec(data_spec, g))
                shared_invvar_list.append(_slice_spec(invvar_spec, g))
                shared_pixg_list.append(_slice_spec(pixg_spec, g))
        else:
            for bn in bandnames_with_psf:
                data, invvar, pixg = per_band_full[bn]
                d_g, i_g, p_g = slice_band_image(data, invvar, pixg, g, copy=False)
                data_list.append(d_g)
                invvar_list.append(i_g)
                pixg_list.append(p_g)

        halo_indices = np.setdiff1d(
            g.halo_source_indices, g.source_indices, assume_unique=False,
        ).astype(np.int64)
        fit_indices = np.concatenate([g.source_indices, halo_indices]).astype(np.int64)
        owned_source_mask = np.zeros(fit_indices.shape[0], dtype=bool)
        owned_source_mask[: g.source_indices.shape[0]] = True

        sx = xpix[fit_indices]
        sy = ypix[fit_indices]
        spe = pos_err[fit_indices]

        # position_solver "auto": resolved PER GRID -- a grid whose fit set
        # contains any extended source (EXP/DEV/SERSIC) takes the varpro
        # joint step (captures flux-position-shape coupling, fits free-n);
        # star-only grids keep the fast (and bit-gated) per_source path.
        # MEMORY GUARD (measured on T08285, 2026-07-04): the varpro residual
        # keeps the per-band inner-solve intermediates live across its band
        # loop for the CG jvp/vjp, so peak GPU memory scales like
        # n_src * k_max * H_psf^2 * n_bands. At 1500 src x 32 x 55^2 x 20
        # bands that is ~13 GiB -> every grid OOM'd on 24 GB cards. Grids
        # whose estimate exceeds the budget fall back to per_source
        # (extended sources still fit shapes via the mixed Newton path for
        # EXP; DEV/SERSIC shapes+n stay at seeds there -- logged). Future
        # lift: jax.checkpoint / lax.map over bands in the varpro residual
        # would cut this ~n_bands-fold.
        if position_solver == "auto":
            grid_has_extended = bool(
                np.any(source_type[fit_indices] != SOURCE_STAR))
            k_max_guess = 32.0  # DEFAULT_K_MAX (conservative upper bound)
            tier_w = float(max(psf_shape))
            varpro_mem_gib = (
                len(fit_indices) * k_max_guess * tier_w * tier_w
                * len(bandnames_with_psf) * 4.0 * 1.3 / 2**30
            )
            varpro_fits_memory = varpro_mem_gib <= 8.0
            if grid_has_extended and not varpro_fits_memory:
                logger.info(
                    "grid_gpu: grid %d auto-solver: varpro est %.1f GiB > "
                    "8 GiB budget -> per_source fallback (extended shapes: "
                    "EXP fit, DEV/SERSIC frozen at seeds)",
                    g.grid_id, varpro_mem_gib,
                )
            task_position_solver = (
                "joint_varpro" if (grid_has_extended and varpro_fits_memory)
                else "per_source")
            auto_solver_counts[task_position_solver] += 1
        else:
            # Island mode: the per-grid passes are S1-only (0 joint
            # iterations), so no position step ever runs in the grid
            # workers; hand the task a valid solver name for the
            # runner/solver plumbing.
            task_position_solver = (
                "per_source" if island_mode else position_solver)

        task = PerGridTask(
            grid_id=g.grid_id, grid_row=g.grid_row, grid_col=g.grid_col,
            bandnames=list(bandnames_with_psf),
            data_per_band=data_list,
            invvar_per_band=invvar_list,
            pix_good_per_band=pixg_list,
            psfs_per_band=psfs_per_band_objs,
            # Per-grid stamp tier: largest requirement among this grid's fit
            # sources (>= native PSF size). The runner pads the PSF stamp.
            psf_shape=(
                (int(src_tier[fit_indices].max()),) * 2
                if len(fit_indices) else psf_shape
            ),
            grid_x0=g.x0_pad, grid_y0=g.y0_pad,
            source_indices=fit_indices,
            pos_x_init=sx, pos_y_init=sy, pos_err_pix=spe,
            source_type=source_type[fit_indices].astype(np.int32),
            logre_init=logre_init[fit_indices].astype(np.float32),
            ee1_init=ee1_init[fit_indices].astype(np.float32),
            ee2_init=ee2_init[fit_indices].astype(np.float32),
            sersic_n_init=sersic_n_init[fit_indices].astype(np.float32),
            render_backend=render_backend,
            pos_max_shift_pix=pos_max_shift_pix,
            bg_x_origin=bg_x_origin, bg_x_scale=bg_x_scale,
            bg_y_origin=bg_y_origin, bg_y_scale=bg_y_scale,
            bg_poly_order=bg_poly_order,
            single_stage=single_stage,
            n_opt_iters=n_opt_iters,
            pos_step_max_pix=pos_step_max_pix,
            dlnp_stop_pix=dlnp_stop_pix,
            position_solver=task_position_solver,
            joint_cg_max_iters=joint_cg_max_iters,
            joint_cg_tol=joint_cg_tol,
            joint_use_preconditioner=joint_use_preconditioner,
            pos_stop_unconverged_frac=pos_stop_unconverged_frac,
            chi2_ftol=chi2_ftol,
            fit_sersic_n=fit_sersic_n,
            n_fit_min_snr=n_fit_min_snr,
            wing_peeling=wing_peeling,
            wing_peel_tier=wing_peel_tier,
            wing_peel_max_sources=wing_peel_max_sources,
            shared_data_per_band=shared_data_list,
            shared_invvar_per_band=shared_invvar_list,
            shared_pix_good_per_band=shared_pixg_list,
            owned_source_mask=owned_source_mask,
        )
        tasks.append(task)

    if position_solver == "auto":
        logger.info(
            "grid_gpu: position_solver=auto resolved per grid: "
            "%d grid(s) -> joint_varpro (extended sources), "
            "%d grid(s) -> per_source (star-only)",
            auto_solver_counts["joint_varpro"],
            auto_solver_counts["per_source"],
        )

    # ---- Resolve GPU devices: 1 worker per GPU per task (limited by tasks count) ----
    logger.info("grid_gpu: dispatching %d tasks to %d GPU worker(s) "
                "(devices %s)", len(tasks), n_workers, devices)
    log_file_cfg = (cfg.get("logging", {}) or {}).get("file")
    worker_log_file = None
    if log_file_cfg and str(log_file_cfg).strip().lower() != "auto":
        worker_log_file = str(log_file_cfg)

    # ---- Island-architecture pre-stage (position_solver: island) ----
    # Pass 1 (per-grid S1 seed solve) + the field-level island fit run
    # here; `tasks` is then REPLACED by the pass-2 (S1' final solve)
    # tasks at the island-converged nonlinear parameters, and the
    # standard dispatch + merge + writeback below runs unchanged.
    island_meta = None
    if island_mode:
        from .island_stage import run_island_prestage

        # ---- Catalog-driven Gaussian PRIOR columns (opt-in) ----------
        # A prior exists ONLY where the user supplied its WIDTH sigma
        # column; absent/blank/<=0 -> NaN -> inv_sigma 0 (no prior, pure
        # data-driven). Prior MEAN = the catalog seed itself. POS_ERR is
        # read case-sensitively (Gaia-augmentation writes it); the
        # *_PRIOR_SIG columns are case-insensitive. FLUX_{band}_PRIOR_SIG
        # pulls the fitted amplitude toward the catalog FLUX_{band} (only
        # where that flux seed is itself a valid catalog value), in the
        # same reference-ZP scaled-flux units the CPU prior path uses.
        prior_cmap = case_insensitive_col_map(catalog.columns)

        def _read_sig(primary, alias=None):
            # R2: the island reader accepts the CPU per-patch *_ERR name
            # as an alias for the *_PRIOR_SIG width, with *_PRIOR_SIG
            # taking PRECEDENCE when both are present, so one catalog
            # regularizes consistently under either solver.
            col = prior_cmap.get(primary.lower())
            if col is None and alias is not None:
                col = prior_cmap.get(alias.lower())
            if col is None:
                return np.full(n_total, np.nan, dtype=np.float32)
            v = pd.to_numeric(catalog[col], errors="coerce").to_numpy(dtype=np.float32)
            return np.where(np.isfinite(v) & (v > 0), v, np.nan).astype(np.float32)

        if "POS_ERR" in catalog.columns:
            _pe = pd.to_numeric(catalog["POS_ERR"], errors="coerce").to_numpy(dtype=np.float32)
            pos_err_prior = np.where(
                np.isfinite(_pe) & (_pe > 0), _pe, np.nan).astype(np.float32)
        else:
            pos_err_prior = np.full(n_total, np.nan, dtype=np.float32)
        re_prior_sig = _read_sig("Re_PRIOR_SIG", "Re_ERR")
        ell_prior_sig = _read_sig("ELL_PRIOR_SIG", "ELL_ERR")
        theta_prior_sig = _read_sig("THETA_PRIOR_SIG", "THETA_ERR")
        sersic_n_prior_sig = _read_sig("SERSIC_N_PRIOR_SIG")  # no CPU alias

        # R3: a shape prior is meaningful only when the shape VALUE came
        # from the catalog (matches the CPU *_from_cat guard); a prior
        # width supplied without its value column/cell would anchor to
        # the internal fallback default, so drop it there (+ warn).
        def _val_from_cat(name):
            col = prior_cmap.get(name.lower())
            if col is None:
                return np.zeros(n_total, dtype=bool)
            v = pd.to_numeric(catalog[col], errors="coerce").to_numpy()
            return np.isfinite(v)
        _re_fc = _val_from_cat("Re")
        _ell_fc = _val_from_cat("ELL")
        _th_fc = _val_from_cat("THETA")
        for _sig, _fc, _nm in ((re_prior_sig, _re_fc, "Re"),
                               (ell_prior_sig, _ell_fc, "ELL"),
                               (theta_prior_sig, _th_fc, "THETA")):
            _drop = int(np.sum(np.isfinite(_sig) & ~_fc))
            if _drop:
                logger.warning(
                    "grid_gpu[island]: %d source(s) supply a %s prior width "
                    "but no catalog %s value -- skipping the %s prior there "
                    "(a prior on the fallback default is meaningless).",
                    _drop, _nm, _nm, _nm)
        re_prior_sig = np.where(_re_fc, re_prior_sig, np.nan).astype(np.float32)
        ell_prior_sig = np.where(_ell_fc, ell_prior_sig, np.nan).astype(np.float32)
        theta_prior_sig = np.where(
            _th_fc, theta_prior_sig, np.nan).astype(np.float32)

        # NOTE: per-band FLUX priors (FLUX_{band}_PRIOR_SIG / _ERR) are
        # DEFERRED to the future field-level flux solve (P7 de-gridding).
        # The reported FLUX/FLUXERR come from the data-only pass-2 S1'
        # per-grid solve, which applies no flux prior, so a flux prior on
        # the discarded internal island flux would not affect any reported
        # value. Only the NONLINEAR (position/shape/n) priors are read.
        _any_prior = any(bool(np.any(np.isfinite(a))) for a in (
            pos_err_prior, re_prior_sig, ell_prior_sig, theta_prior_sig,
            sersic_n_prior_sig))
        if _any_prior:
            logger.info(
                "grid_gpu[island]: catalog Gaussian priors active "
                "(POS_ERR=%d, Re=%d, ELL=%d, THETA=%d, SERSIC_N=%d "
                "sources with a valid prior width)",
                int(np.sum(np.isfinite(pos_err_prior))),
                int(np.sum(np.isfinite(re_prior_sig))),
                int(np.sum(np.isfinite(ell_prior_sig))),
                int(np.sum(np.isfinite(theta_prior_sig))),
                int(np.sum(np.isfinite(sersic_n_prior_sig))))

        tasks, island_meta = run_island_prestage(
            tasks=tasks, grids=grids, devices=devices, gpu_cfg=gpu_cfg,
            island_cfg=island_cfg,
            bandnames_with_psf=bandnames_with_psf,
            psfs_per_band_objs=psfs_per_band_objs,
            per_band_full=per_band_full,
            shared_full_specs=shared_full_specs,
            image_hw=(image_h, image_w),
            keep_idx=keep_idx, n_total=n_total,
            xpix=xpix, ypix=ypix,
            source_type=source_type,
            logre_init=logre_init, ee1_init=ee1_init, ee2_init=ee2_init,
            sersic_n_init=sersic_n_init,
            render_backend=render_backend,
            fit_sersic_n=fit_sersic_n, n_fit_min_snr=n_fit_min_snr,
            pos_max_shift_pix=pos_max_shift_pix,
            pos_err_prior=pos_err_prior,
            re_prior_sig=re_prior_sig, ell_prior_sig=ell_prior_sig,
            theta_prior_sig=theta_prior_sig,
            sersic_n_prior_sig=sersic_n_prior_sig,
            bg_x_origin=bg_x_origin, bg_x_scale=bg_x_scale,
            bg_y_origin=bg_y_origin, bg_y_scale=bg_y_scale,
            bg_poly_order=bg_poly_order,
            jit_cache_dir=jit_cache_dir,
            worker_log_file=worker_log_file,
            dispatch_fn=_run_grids_on_slots,
            # effarc2 K7: pass 1 gets the same MEASURED packing (+ MPS +
            # OOM retry) the final grid pass uses, instead of one exclusive
            # worker per GPU. `gpu_cfg`/`ctx`/cache paths are bound here so
            # island_stage stays JAX-free and pipeline-agnostic.
            # DIAGNOSTIC ARM (2026-08-29, NOT a ship candidate): K7 disabled.
            # None sends island pass 1 down the PRE-ARC path -- one exclusive
            # worker per GPU instead of the measured 8/GPU packing. Tests
            # whether that 8x rise in concurrent workers is what cost the engine
            # its pass-A determinism: the pre-arc engine reproduced pass A to
            # 1e-7 (0.0001 mmag over 10 pairs) and gave 91,191 islands EVERY
            # run, while dev3 (43x) and dev4 (97x) both vary. Everything else in
            # this tree is the shipped engine, byte for byte.
            dispatch_packed_fn=None,
            bright_mask_det=state.get("bright_mask_det"),
            work_dir=str(cfg["outputs"]["work_dir"]),
            # v2 (phantom columns) consumes the routed detections at the
            # fit-set level, so the v1 sequential-subtraction path must
            # not see them (double handling).
            bright_mask_sub=(None if phantom_mask is not None
                             else state.get("bright_mask_sub")),
            bright_mask_sub_nsigma=state.get("bright_mask_sub_nsigma"),
            phantom_mask=phantom_mask,
        )
        logger.info("grid_gpu[island]: dispatching pass 2 (S1' final "
                    "solve at converged params) through the standard "
                    "grid dispatch")

    # ---- Post-solve error-model flags (task #27) ---------------------
    # Enable the residual-map confusion measurement ONLY on the tasks of
    # the FINAL solve pass (for island mode `tasks` is now the pass-2
    # S1' set; the pass-1 seed solves above ran with the flag off).
    error_model_cfg = (gpu_cfg.get("error_model", {}) or {})
    confusion_enabled = bool(error_model_cfg.get("confusion_enabled", True))
    # LEG C (task #20): per-band correlated-coadd-noise variance inflation.
    # Applied only to the REPORTED GEOMETRIC (pos/shape/n) errors, as SEPARATE
    # sqrt(F_corr)-scaled columns (the fit weights + fitted values + raw
    # FLUXERR + *_err are untouched). Re-scope 2026-07-08: the flux side is
    # #27-only; F_corr is a GEOMETRIC-error term now.
    correlated_noise_enabled = bool(
        error_model_cfg.get("correlated_noise_enabled", True))
    # LEG D (task #20): per-source reduced-chi2 error rescaling of the
    # GEOMETRIC errors. The nonlinear chi2_nu (chi2_nu_nl) comes from the
    # island stage; there is no flux chi2_nu (removed in the 2026-07-08
    # re-scope -- the flexible per-source fit absorbs crowded flux confusion,
    # which task #27's annulus sigma_conf captures instead).
    chi2_nu_enabled = bool(error_model_cfg.get("chi2_nu_enabled", True))
    # Enable the worker residual pass (error_model.py) on the FINAL solve
    # tasks when a consumer needs it: #27 flux confusion (sigma_conf) or LEG C
    # crowded-robust GEOMETRIC residual F_corr. One render serves both. When
    # neither is on, no residual pass runs -> existing columns byte-identical.
    for t in tasks:
        if confusion_enabled:
            t.compute_confusion = True
        if correlated_noise_enabled:
            t.compute_fcorr_resid = True
    # Island nonlinear-parameter errors (task #28a): the measurement
    # itself runs inside the island stage above (run_island_prestage
    # reads the same knob); this flag only gates the output columns.
    param_errors_enabled = bool(
        error_model_cfg.get("param_errors_enabled", True))
    # #17: per-source model-quality columns (rchisq/fracflux/fracin/
    # fracmasked) are emitted ONLY when island mode is active AND the knob is
    # on; the metric itself is computed in the island stage (which reads the
    # same knob). This flag only gates the output columns so an off run is
    # byte-identical (columns absent).
    source_metrics_enabled = bool(
        error_model_cfg.get("source_metrics_enabled", True))
    emit_source_metrics = bool(
        island_meta is not None and source_metrics_enabled
        and island_meta.get("source_metrics_active", False))
    # D2 (root-fix D): SHAPE-marginalized flux-error columns FLUXERR_MARG* /
    # MAGERR_MARG* are emitted ONLY in island mode when the knob is on; the
    # per-band marginalization INCREMENT is computed in the island stage
    # (which reads the same knob). This flag only gates the output columns
    # so an off run is byte-identical (columns absent).
    marginal_flux_err_enabled = bool(
        error_model_cfg.get("marginal_flux_err_enabled", True))
    emit_flux_marg = bool(
        island_meta is not None and marginal_flux_err_enabled
        and island_meta.get("flux_marg_active", False))
    # R1: prior_frac disclosure columns are emitted ONLY when island mode
    # is active AND the catalog supplied priors AND param errors are on
    # (the nonlinear prior_frac needs the post-convergence Hessian). This
    # mirrors the param_errors_enabled column gating so a no-prior run is
    # byte-identical -- the columns are absent entirely.
    emit_prior_frac = bool(
        island_meta is not None and param_errors_enabled
        and island_meta.get("priors_active", False))

    # ---- Run per-grid workers (adaptive, memory-safe packing) ----
    # workers_per_gpu is chosen ADAPTIVELY from probed free GPU memory + a
    # conservative per-grid peak estimate, so packing is safe on ANY machine
    # and for ANY image/catalog (large grids on a 24GB card -> 1/GPU; small
    # grids or big GPUs -> pack). Packed workers (N>1) get a HARD per-worker
    # BFC cap (~0.9/N of the card) so they cannot collectively OOM; any grid
    # that still exceeds its cap is RETRIED on an exclusive GPU. Per-grid math
    # is deterministic + independent (merged by source_indices), so the merged
    # result is identical regardless of packing -- only scheduling changes.
    # Override: env TRACT7DT_GPU_WORKERS_PER_GPU or compute.gpu.workers_per_gpu.
    wpg, alloc_mode, per_worker_mem, wpg_diag = _choose_workers_per_gpu(tasks, devices, gpu_cfg)
    logger.info("grid_gpu: %s", wpg_diag)
    # Position-step chunking is only worth it when PACKING (>1 worker/GPU): it
    # cuts per-grid peak so co-located workers fit, but adds overhead (a cold
    # JIT of the lax.map graph) with NO benefit at 1 worker/GPU. So enable it
    # for the workers ONLY when packing; otherwise force the fast, bit-exact
    # single-vmap path. Workers (spawn) inherit this env at import; the wpg
    # decision above already assumed chunking (so large-grid fields could pack).
    # A user value of 0 stays 0 (chunking fully disabled).
    _cfg_chunk = int(os.environ.get("TRACT7DT_POS_STEP_CHUNK", "512"))
    os.environ["TRACT7DT_POS_STEP_CHUNK"] = str(_cfg_chunk if (wpg > 1 and _cfg_chunk > 0) else 0)
    # The MIXED (extended-source) Newton's memory chunking is a SURVIVAL
    # requirement on crowded extended fields (the unchunked per-neighbour
    # gather is tens of GB at 1500 src x 20 bands x k_local~190), not a
    # packing optimization -- so its chunk-size cap is passed separately
    # and survives the wpg=1 zeroing above. A user-explicit
    # TRACT7DT_POS_STEP_CHUNK=0 still disables both paths.
    os.environ["TRACT7DT_SRC_STEP_CHUNK_CAP"] = str(_cfg_chunk)
    logger.info("grid_gpu: position-step chunk size for workers = %s "
                "(wpg=%d; mixed-path cap = %s)",
                os.environ["TRACT7DT_POS_STEP_CHUNK"], wpg,
                os.environ["TRACT7DT_SRC_STEP_CHUNK_CAP"])
    ctx = mp.get_context("spawn")
    disable_worker_warmup = os.environ.get("TRACT7DT_BENCH_DISABLE_WORKER_WARMUP") == "1"
    n_slots = len(devices) * wpg
    # Tell every worker how many peers it shares the machine with, so the
    # host-side band pools in error_model divide the cores/RAM instead of each
    # worker claiming all of them (T2).
    for t in tasks:
        t.n_grid_slots = int(n_slots)
    results: list[PerGridResult] = []

    # R17 (2026-09-26): a single slot also runs in a spawned worker. The old
    # solo-inline branch solved the grids in THIS process, whose JAX was
    # imported without x64, so its _worker_init refused to run (every 1-GPU
    # wpg=1 or one-grid run died after the island stage). A worker gets x64
    # before its JAX import, exactly as for packed slots and pass 1.
    #
    # MPS (default on) lets the packed workers run concurrent GPU kernels
    # instead of time-slicing. Only meaningful when packing (wpg>1). Private per-user
    # daemon, started before workers spawn (so they inherit the pipe dir),
    # stopped after. Disable via compute.gpu.use_mps:false or env
    # TRACT7DT_GPU_USE_MPS=0.
    use_mps = (str(os.environ.get("TRACT7DT_GPU_USE_MPS",
                   str(gpu_cfg.get("use_mps", True)))).strip().lower()
               not in ("0", "false", "no", "off")) and wpg > 1
    mps_active, mps_started, mps_pipe = _maybe_start_mps(use_mps)
    slot_devices = [devices[s % len(devices)] for s in range(n_slots)]
    teardown_thread: threading.Thread | None = None
    try:
        results, teardown_thread = _run_grids_on_slots(
            tasks, slot_devices, ctx, alloc_mode, per_worker_mem,
            jit_cache_dir, worker_log_file, disable_worker_warmup,
            defer_shutdown=True,
        )
        # OOM safety net: any grid that exceeded its per-worker cap under
        # packing is re-run 1 worker/GPU with the full (platform) allocator,
        # guaranteeing a correct, bit-identical result (never NaN).
        oom = [r for r in results if _is_oom_error(r)]
        if oom:
            by_grid = {t.grid_id: t for t in tasks}
            retry_tasks = [by_grid[r.grid_id] for r in oom if r.grid_id in by_grid]
            logger.warning(
                "grid_gpu: %d grid(s) OOM'd under %dw/GPU packing; retrying "
                "1 worker/GPU (exclusive memory): grids=%s",
                len(retry_tasks), wpg, sorted(r.grid_id for r in oom),
            )
            if teardown_thread is not None:
                # The retry pass needs the packed workers' GPU arenas
                # fully released before its exclusive workers start.
                teardown_thread.join()
                teardown_thread = None
            # The retry runs exclusive (1 worker/GPU), so the host-side
            # band pools get a correspondingly larger share.
            for t in retry_tasks:
                t.n_grid_slots = len(devices)
            retry_results, _ = _run_grids_on_slots(
                retry_tasks, list(devices), ctx, "platform",
                float(gpu_cfg.get("mem_fraction", 0.5)),
                jit_cache_dir, worker_log_file, disable_worker_warmup,
            )
            rmap = {r.grid_id: r for r in retry_results}
            results = [rmap.get(r.grid_id, r) if _is_oom_error(r) else r for r in results]
    finally:
        _cleanup_grid_shm()
        if teardown_thread is not None:
            # Teardown is still draining in the background (~14s of BFC
            # arena releases): quit MPS only after the last client is
            # gone, off the critical path. Non-daemon thread, so even a
            # post-merge crash cannot leak the daemon past process exit
            # (the atexit backstop also remains in place).
            def _late_stop(t=teardown_thread, started=mps_started, pipe=mps_pipe):
                t.join()
                _stop_mps(started, pipe)

            threading.Thread(
                target=_late_stop, name="tract7dt-mps-late-stop",
            ).start()
        else:
            _stop_mps(mps_started, mps_pipe)

    # Never ship degraded science (R12): a grid that still fails after the
    # retry -- including an error-model failure inside it (runner) -- fails
    # the run (RUN_STATUS.json "failed", non-zero exit). It used to be logged
    # and its sources written as NaN rows while the run exited 0.
    run_status.note("final_pass", {
        "n_grids": len(tasks), "devices": [int(d) for d in devices],
        "workers_per_gpu": int(wpg), "n_slots": int(n_slots)})
    _raise_if_grids_failed(results)

    # ---- LEG C: per-band correlated-coadd-noise factor F_corr (task #20) ---
    # TWO estimators, per band: (1) RAW-SKY -- measured from each coadd's own
    # source-free sky (pure numpy, no GPU); clean on SPARSE fields but NULLS
    # to 1.0 on crowded ones (unresolved-source structure contaminates the
    # "sky" ACF). (2) RESIDUAL -- measured in the workers from the
    # model-subtracted residual's LEAST-CROWDED tiles (crowded-robust; the
    # resampling correlation is field-uniform so a clean-tile estimate applies
    # field-wide), combined across grids here. PREFER the residual estimate;
    # fall back to raw-sky; then to 1.0 (no inflation). The final per-band
    # F_corr is a SCALAR variance inflation applied ONLY to the reported
    # errors below. img_scaled / bad survive here. Read-only on the fit state.
    fcorr_by_band: dict[str, float] = {}
    if correlated_noise_enabled:
        _fc_t0 = time.perf_counter()
        # Band-parallel raw-sky pass (see _fcorr_raw_sky_pass for why threads,
        # and for the determinism contract on log text + diag order).
        _fc_bands = [bn for bn in bandnames_with_psf
                     if (image_dict.get(bn) or {}).get("img_scaled") is not None]
        fcorr_raw_by_band, _fc_diags = _fcorr_raw_sky_pass(
            _fc_bands, image_dict, psfs)

        # ---- Combine the per-grid RESIDUAL F_corr (crowded-robust) ----
        # Each grid returns res.fcorr_resid, a list over fit-band index of
        # (F_corr, n_clean, fallback) or None. F_corr is field-uniform, so
        # combine across grids by a clean-pixel-weighted mean of the valid
        # (non-fallback, finite, > 1) per-grid estimates.
        fcorr_resid_by_band: dict[str, float] = {}
        _resid_nclean: dict[str, int] = {}
        for res in results:
            fr = getattr(res, "fcorr_resid", None)
            if not fr:
                continue
            for bi_fit, bn in enumerate(bandnames_with_psf):
                if bi_fit >= len(fr) or fr[bi_fit] is None:
                    continue
                Fv, nc, fb = fr[bi_fit]
                if fb is not None or not np.isfinite(Fv) or Fv <= 1.0 or nc <= 0:
                    continue
                acc = fcorr_resid_by_band.get(bn, 0.0)
                fcorr_resid_by_band[bn] = acc + float(Fv) * int(nc)
                _resid_nclean[bn] = _resid_nclean.get(bn, 0) + int(nc)
        for bn in list(fcorr_resid_by_band.keys()):
            nc = _resid_nclean.get(bn, 0)
            fcorr_resid_by_band[bn] = (fcorr_resid_by_band[bn] / nc
                                       if nc > 0 else np.nan)

        # ---- Choose the final per-band F_corr: residual > raw-sky > 1 ----
        _fc_source: dict[str, str] = {}
        for bn in bandnames_with_psf:
            f_res = fcorr_resid_by_band.get(bn, np.nan)
            f_raw = fcorr_raw_by_band.get(bn, np.nan)
            if np.isfinite(f_res) and f_res > 1.0:
                fcorr_by_band[bn] = float(f_res)
                _fc_source[bn] = "residual"
            elif np.isfinite(f_raw) and f_raw > 1.0:
                fcorr_by_band[bn] = float(f_raw)
                _fc_source[bn] = "raw-sky"
            else:
                fcorr_by_band[bn] = 1.0
                _fc_source[bn] = "none(=1)"
        _n_res = sum(1 for v in _fc_source.values() if v == "residual")
        _n_raw = sum(1 for v in _fc_source.values() if v == "raw-sky")
        logger.info(
            "grid_gpu: correlated-noise F_corr chosen for %d bands in %.2fs "
            "(median F_corr=%.2f; %d residual, %d raw-sky, %d none)",
            len(fcorr_by_band), time.perf_counter() - _fc_t0,
            float(np.median(list(fcorr_by_band.values())))
            if fcorr_by_band else 1.0, _n_res, _n_raw,
            len(fcorr_by_band) - _n_res - _n_raw)
        # Persist the per-band factors next to the outputs (disclosure).
        try:
            work_dir = Path(cfg["outputs"]["work_dir"])
            _fc_rows = []
            for dg in _fc_diags:
                bn = dg.get("band")
                _fc_rows.append({
                    "band": bn,
                    "F_corr": fcorr_by_band.get(bn, np.nan),
                    "F_corr_source": _fc_source.get(bn, ""),
                    "sigma_inflation": float(np.sqrt(
                        max(fcorr_by_band.get(bn, 1.0), 0.0))),
                    "F_corr_residual": fcorr_resid_by_band.get(bn, np.nan),
                    "F_corr_resid_nclean": _resid_nclean.get(bn, 0),
                    "F_corr_rawsky": fcorr_raw_by_band.get(bn, np.nan),
                    "F_corr_acf": dg.get("F_corr_acf"),
                    "F_corr_box_xcheck": dg.get("F_corr_box"),
                    "rho_1px": dg.get("rho_1px"),
                    "acf_shape": dg.get("acf_shape"),
                    "acf_q2": dg.get("acf_q2"),
                    "sky_frac": dg.get("sky_frac"),
                    "fallback": dg.get("fallback"),
                })
            if _fc_rows:
                pd.DataFrame(_fc_rows).to_csv(
                    work_dir / "correlated_noise_factors.csv", index=False)
        except Exception as e:
            logger.warning("grid_gpu: could not write "
                           "correlated_noise_factors.csv (%s)", str(e)[:120])

    # ---- Merge per-grid results into output_catalog.csv ----
    t0 = time.perf_counter()
    out = catalog.copy()
    # Add fit columns matching per-patch output schema.
    out["x_pix_white_fit"] = np.nan
    out["y_pix_white_fit"] = np.nan
    out["RA_fit"] = np.nan
    out["DEC_fit"] = np.nan
    out["opt_converged"] = False
    out["position_stranded"] = False
    out["opt_niters"] = 0
    out["opt_hit_max_iters"] = False
    out["opt_last_dlnp"] = np.nan
    for bn in bandnames:
        out[f"FLUX_{bn}_fit"] = np.nan
        out[f"FLUXERR_{bn}_fit"] = np.nan
    if confusion_enabled:
        # Measured confusion noise (error_model.py) + total error columns.
        # FLUXERR_* keeps its exact legacy meaning (marginal statistical
        # error of the linear solve); FLUXERR_TOT_* adds sigma_conf in
        # quadrature and is filled after the per-grid merge below. This #27
        # pair IS the entire flux error budget (re-scope 2026-07-08 -- the
        # flux LEG C/D columns FLUXERR_CORR_* / chi2_nu_flux_* /
        # FLUXERR_TOT_NEW_* were removed; F_corr + chi2_nu now apply ONLY to
        # the geometric errors below).
        for bn in bandnames:
            out[f"FLUXERR_CONF_{bn}_fit"] = np.nan
    out["grid_id_fit"] = -1
    out["stype_fit"] = ""
    for col in ("re_pix_fit", "ab_fit", "phi_deg_fit", "Re_fit", "ELL_fit", "THETA_fit",
                "sersic_n_fit", "sersic_n_err"):
        out[col] = np.nan
    if island_meta is not None and param_errors_enabled:
        # Island nonlinear-parameter 1-sigma errors (task #28a); filled
        # from island_meta in the island block below. The off state
        # omits the columns entirely (FLUXERR_CONF gating idiom).
        for col in ("x_pix_white_err", "y_pix_white_err",
                    "Re_err", "ELL_err", "THETA_err"):
            out[col] = np.nan
        if correlated_noise_enabled:
            # LEG C: correlated-noise-inflated nonlinear-parameter errors,
            # *_err_corr = sqrt(F_corr_eff) * *_err, where F_corr_eff is the
            # source's SNR^2-weighted (~ per-band position/shape Hessian)
            # mean of the per-band F_corr (the island fit is joint-multiband).
            # SEPARATE columns; the raw *_err are untouched. sersic_n_err_corr
            # is created unconditionally below since sersic_n_err always exists.
            for col in ("x_pix_white_err_corr", "y_pix_white_err_corr",
                        "Re_err_corr", "ELL_err_corr", "THETA_err_corr"):
                out[col] = np.nan
        if chi2_nu_enabled:
            # LEG D: combined C x D total nonlinear errors, *_err_tot =
            # *_err * sqrt(F_corr_eff) * sqrt(max(1, chi2_nu_nl)). SEPARATE
            # columns; raw *_err untouched. chi2_nu_nl is the per-island
            # reduced chi2 (shared by members).
            for col in ("x_pix_white_err_tot", "y_pix_white_err_tot",
                        "Re_err_tot", "ELL_err_tot", "THETA_err_tot"):
                out[col] = np.nan
    if correlated_noise_enabled:
        # sersic_n_err always exists (created above); its corr variant too.
        out["sersic_n_err_corr"] = np.nan
    if chi2_nu_enabled:
        # sersic_n_err_tot (C x D) + the per-island chi2_nu_nl diagnostic.
        out["sersic_n_err_tot"] = np.nan
        out["chi2_nu_nl"] = np.nan
    if emit_prior_frac:
        # R1 prior_frac disclosure (island mode + priors active + param
        # errors on). Additive: absent entirely when no priors, so a
        # no-prior run is byte-identical. Filled in the island block.
        for col in ("x_pix_white_prior_frac", "y_pix_white_prior_frac",
                    "Re_prior_frac", "ELL_prior_frac", "THETA_prior_frac",
                    "sersic_n_prior_frac"):
            out[col] = np.nan
    # sersic_n_err: filled from the per-grid results below for FREE-n rows
    # (diagonal / flux-marginalized 1-sigma from the varpro residual jvp at
    # convergence -- see solver_grid._varpro_sersic_n_err); frozen rows and
    # non-varpro grids stay NaN.
    # True where TYPE==SERSIC and the Sersic index was NOT freed (fit off,
    # non-varpro solver, or S1 S/N below compute.gpu.galaxy.n_fit_min_snr).
    out["sersic_n_frozen"] = False
    # shape_frozen (seed->output honesty): True for a DEV/SERSIC row whose
    # grid ran a shape-freezing solver (per_source/joint_lsqr), so its
    # Re_fit/ELL_fit/THETA_fit are the catalog seed rather than a fit.
    # Informational only (does NOT flip fit_is_reliable): the flux is a
    # real measurement conditioned on the assumed shape, exactly like a
    # frozen-n flux. Filled from the per-grid results below; island mode
    # overrides it (islands refit shapes).
    out["shape_frozen"] = False
    fit_logre_all = logre_init.astype(np.float32).copy()
    fit_ee1_all = ee1_init.astype(np.float32).copy()
    fit_ee2_all = ee2_init.astype(np.float32).copy()

    # Write per-source fit results from each grid's owned indices.
    # IMPORTANT: PerGridResult.fluxes is indexed by position in
    # `bandnames_with_psf` (which excludes any band that had < 10
    # ePSF cells and was dropped). We need to map each FIT-axis
    # position back to its bandname.
    wcs_white = next(iter(image_dict.values()))["wcs"]
    # blend_resolve R3: pass-2 sentinel star columns (negative source
    # indices, injected by the island stage so a resolved star's light is
    # MODELED in the S1' flux re-solve instead of re-absorbed into its
    # host). They are solve-only: capture their solved per-band fluxes for
    # blend_resolved.csv, then strip the rows so no writeback (pandas .loc
    # with a negative label would silently create/overwrite rows) ever
    # sees them.
    _sentinel_flux: dict[int, np.ndarray] = {}

    def _strip_sentinels(res):
        idx0 = np.asarray(res.source_indices, np.int64)
        sent = idx0 < 0
        if not np.any(sent):
            return res
        for _sq in np.nonzero(sent)[0]:
            _sentinel_flux[int(idx0[_sq])] = np.asarray(
                res.fluxes[_sq], dtype=np.float64)
        keepm = ~sent
        rep = {}
        for _fld in ("pos_x", "pos_y", "logre", "ee1", "ee2", "fluxes",
                     "flux_var", "pos_var", "source_indices",
                     "position_stranded", "sersic_n_fit",
                     "sersic_n_frozen", "sersic_n_err", "shape_frozen",
                     "fluxerr_conf"):
            _v = getattr(res, _fld, None)
            if (_v is not None and isinstance(_v, np.ndarray)
                    and _v.shape[:1] == idx0.shape):
                rep[_fld] = _v[keepm]
        rep["n_sources"] = int(keepm.sum())
        import dataclasses as _dc4
        return _dc4.replace(res, **rep)

    results = [_strip_sentinels(r) if not getattr(r, "error", None) else r
               for r in results]
    # R09: record the b_hat actually subtracted, per grid x band. Written on
    # every run (header only when nothing was subtracted) so a reused
    # work_dir never keeps a stale table.
    _bhat_df = blank_response_bhat_table(results, bandnames_with_psf)
    _bhat_path = Path(cfg["outputs"]["work_dir"]) / "blank_response_bhat.csv"
    _bhat_df.to_csv(_bhat_path, index=False)
    logger.info("grid_gpu: blank-response b_hat table written: %s "
                "(%d grid-band rows)", _bhat_path, len(_bhat_df))
    for res in results:
        idx = res.source_indices
        # A failed grid's placeholder carries the catalog seeds as positions;
        # written here they would pass as fitted. _raise_if_grids_failed ended
        # the run before this point (R12).
        assert not getattr(res, "error", None), res.grid_id
        out.loc[idx, "x_pix_white_fit"] = res.pos_x
        out.loc[idx, "y_pix_white_fit"] = res.pos_y
        try:
            ra_fit, dec_fit = wcs_white.all_pix2world(res.pos_x, res.pos_y, 0)
            out.loc[idx, "RA_fit"] = ra_fit
            out.loc[idx, "DEC_fit"] = dec_fit
        except Exception:
            pass
        out.loc[idx, "opt_converged"] = bool(res.s3_converged)
        out.loc[idx, "opt_niters"] = int(res.n_outer_iters)
        # PER-SOURCE hit_max_iters: when a grid runs to the iteration ceiling
        # because of a few oscillating stragglers, only THOSE sources (the
        # position_stranded mask at the last iteration) lose reliability --
        # previously the grid-level bool poisoned fit_is_reliable for every
        # source sharing the grid (observed: 1 straggler flagged all 220).
        stranded_arr = (
            np.asarray(res.position_stranded, dtype=bool)
            if getattr(res, "position_stranded", None) is not None
            else np.zeros(len(idx), dtype=bool)
        )
        out.loc[idx, "opt_hit_max_iters"] = (
            stranded_arr if not res.s3_converged
            else np.zeros(len(idx), dtype=bool)
        )
        if getattr(res, "position_stranded", None) is not None:
            out.loc[idx, "position_stranded"] = stranded_arr
        out.loc[idx, "grid_id_fit"] = int(res.grid_id)
        res_source_type = source_type[idx]
        stype_names = np.full(len(idx), "star", dtype=object)
        stype_names[res_source_type == SOURCE_EXP] = "exp"
        stype_names[res_source_type == SOURCE_DEV] = "dev"
        stype_names[res_source_type == SOURCE_SERSIC] = "sersic"
        out.loc[idx, "stype_fit"] = stype_names
        fit_logre_all[idx] = np.asarray(res.logre, dtype=np.float32)
        fit_ee1_all[idx] = np.asarray(res.ee1, dtype=np.float32)
        fit_ee2_all[idx] = np.asarray(res.ee2, dtype=np.float32)
        exp_local = res_source_type != SOURCE_STAR
        if np.any(exp_local):
            exp_idx = idx[exp_local]
            re_fit, ab_fit, phi_fit = ellipse_soft_to_re_ab_phi_np(
                np.asarray(res.logre)[exp_local],
                np.asarray(res.ee1)[exp_local],
                np.asarray(res.ee2)[exp_local],
            )
            out.loc[exp_idx, "re_pix_fit"] = re_fit
            out.loc[exp_idx, "ab_fit"] = ab_fit
            out.loc[exp_idx, "phi_deg_fit"] = phi_fit
            out.loc[exp_idx, "Re_fit"] = re_fit
            out.loc[exp_idx, "ELL_fit"] = 1.0 - ab_fit
            out.loc[exp_idx, "THETA_fit"] = phi_fit - 90.0
            # Sersic index, reported for DEV/SERSIC rows (EXP renders its
            # fixed profile). Stage-2: the varpro solver FITS n for
            # TYPE=SERSIC rows above the S/N gate -- use the grid's fitted
            # values when present; otherwise (per_source grids, old
            # results) fall back to the catalog seeds.
            n_local = res_source_type[exp_local]
            if getattr(res, "sersic_n_fit", None) is not None:
                n_vals = np.asarray(res.sersic_n_fit, dtype=np.float32)[exp_local]
            else:
                n_vals = sersic_n_init[exp_idx].astype(np.float32)
            keep_n = (n_local == SOURCE_DEV) | (n_local == SOURCE_SERSIC)
            out.loc[exp_idx[keep_n], "sersic_n_fit"] = n_vals[keep_n]
        if getattr(res, "sersic_n_frozen", None) is not None:
            out.loc[idx, "sersic_n_frozen"] = np.asarray(
                res.sersic_n_frozen, dtype=bool)
        if getattr(res, "shape_frozen", None) is not None:
            out.loc[idx, "shape_frozen"] = np.asarray(
                res.shape_frozen, dtype=bool)
        if getattr(res, "sersic_n_err", None) is not None:
            # Free-n rows carry a finite diagonal/flux-marginalized
            # 1-sigma n error; frozen rows keep NaN.
            nerr = np.asarray(res.sersic_n_err, dtype=np.float32)
            good_err = np.isfinite(nerr)
            if np.any(good_err):
                out.loc[idx[good_err], "sersic_n_err"] = nerr[good_err]
        for bi_fit, bn in enumerate(bandnames_with_psf):
            if bi_fit >= res.fluxes.shape[1]:
                continue
            out.loc[idx, f"FLUX_{bn}_fit"] = res.fluxes[:, bi_fit]
            fv = np.asarray(res.flux_var[:, bi_fit])
            fluxerr = np.where(np.isfinite(fv) & (fv > 0), np.sqrt(fv), np.nan)
            out.loc[idx, f"FLUXERR_{bn}_fit"] = fluxerr.astype(np.float32)
            if confusion_enabled and getattr(res, "fluxerr_conf", None) is not None:
                out.loc[idx, f"FLUXERR_CONF_{bn}_fit"] = (
                    np.asarray(res.fluxerr_conf[:, bi_fit], dtype=np.float32))

    # ---- Quality flags so the downstream ZP calibration can run ----
    # Compute `flag_bound_stalled` so the shared reliability function can
    # see when a source's fitted position is pinned at the per-source
    # position bound (i.e., the optimizer wanted to move further but was
    # clipped). The CPU per-patch path produces this column from Tractor's
    # ConstrainedOptimizer; we reproduce the same signal here by detecting
    # post-fit positions within ``stall_epsilon`` of either +-pos_max_shift.
    stall_epsilon = 1.0e-4  # px; bound-distance tolerance for "pinned at limit"
    x_init = pd.to_numeric(out["x_pix_white"], errors="coerce").to_numpy(dtype=float)
    y_init = pd.to_numeric(out["y_pix_white"], errors="coerce").to_numpy(dtype=float)
    x_fit_v = pd.to_numeric(out["x_pix_white_fit"], errors="coerce").to_numpy(dtype=float)
    y_fit_v = pd.to_numeric(out["y_pix_white_fit"], errors="coerce").to_numpy(dtype=float)
    dx = x_fit_v - x_init
    dy = y_fit_v - y_init
    drift_finite = np.isfinite(dx) & np.isfinite(dy)
    near_x_bound = drift_finite & (np.abs(np.abs(dx) - pos_max_shift_pix) < stall_epsilon)
    near_y_bound = drift_finite & (np.abs(np.abs(dy) - pos_max_shift_pix) < stall_epsilon)
    out["flag_bound_stalled"] = (near_x_bound | near_y_bound).astype(bool)

    # Shape-bound analogue: an extended source whose fitted Re is pinned at
    # the solver's Re bound (0.75x the model stamp width; see runner.py) has
    # an unconstrained size -- typically an LSB giant whose chi^2 is flat.
    # Leg-3 forensics (SXDS cross-match) found such sources also carry >1"
    # position drift while passing every other QA flag. Column-only signal:
    # reliability semantics are unchanged (validated separately).
    import pandas as _pd
    re_upper_native = 0.75 * float(psf_stamp_size)
    re_fit_v = _pd.to_numeric(out["re_pix_fit"], errors="coerce").to_numpy(dtype=float)
    # TIER-AWARE ceiling: in island mode the fitted-Re bound is 0.75*TIER,
    # where the tier is chosen from the SEED shape (island_plan). For a
    # promoted tier (111/221) that ceiling is 0.75*83 / 0.75*166, LARGER
    # than the scalar native 0.75*55 -- so comparing island Re to the
    # native ceiling would both MISS genuine bound-pins in promoted-tier
    # islands (a seed-tier-limited Re reported as a measurement) AND
    # false-positive a data-good Re~41 there. Use the per-source island
    # ceiling where defined; fall back to the native ceiling for
    # legacy/per_source rows and where the island bound is undefined
    # (off-island / giants). Bit-identical to the old test when island_meta
    # is None (re_upper_eff == re_upper_native everywhere).
    re_upper_eff = np.full(len(out), re_upper_native, dtype=float)
    if island_meta is not None and "re_upper_pix" in island_meta:
        isl_upper = np.asarray(island_meta["re_upper_pix"], dtype=float)
        has_bound = np.isfinite(isl_upper) & (isl_upper > 0)
        re_upper_eff = np.where(has_bound, isl_upper, re_upper_eff)
    out["flag_shape_bound"] = (
        np.isfinite(re_fit_v)
        & (np.abs(re_fit_v - re_upper_eff) < 1e-2 * re_upper_eff)
    ).astype(bool)

    # ---- Island-mode bookkeeping (before the reliability flag) -------
    # The pass-2 (S1') grid results above carry no nonlinear-solver
    # convergence information (0 joint iterations by construction), so
    # the solver flags are overridden with the ISLAND stage's per-source
    # convergence; four island columns are appended. Existing fit
    # columns keep their names/semantics.
    if island_meta is not None:
        fitm = np.asarray(island_meta["fitted_mask"], dtype=bool)
        conv = np.asarray(island_meta["converged"], dtype=bool)
        gidx = np.where(fitm)[0]
        out.loc[gidx, "opt_converged"] = conv[fitm]
        out.loc[gidx, "opt_niters"] = island_meta["n_steps"][fitm].astype(int)
        out.loc[gidx, "opt_hit_max_iters"] = ~conv[fitm]
        out.loc[gidx, "position_stranded"] = ~conv[fitm]
        out["island_id"] = island_meta["island_id"]
        out["island_nsrc"] = island_meta["island_nsrc"]
        out["island_giant"] = island_meta["island_giant"]
        out["island_stalled"] = island_meta["island_stalled"]
        # #35: frozen-proposal-stall disclosure (clip-saturated runaway
        # class; subset of island_stalled under the default
        # stall_count_mode). Soft fit_quality_issues tag only -- never
        # enters fit_is_reliable (merge.compute_reliability_flag).
        out["island_diverged"] = island_meta["island_diverged"]
        # Stall-yield (2026-07-27): probability-convergence certificate
        # for stalled islands (Tractor-style dlnp semantics). The float
        # column is the best available damped-step chi2 improvement over
        # the full lambda ladder at the stalled state (negative =
        # improvement exists; NaN on non-stalled rows). island_plateau =
        # stalled AND no step improves chi2 by more than the statistical
        # quantum 2*plateau_dlnp_tol (dchi2 = 1 <=> a 1-sigma
        # single-parameter move) => the fit stopped because it is DONE,
        # not stuck. Both are SOFT disclosure columns -- fit_is_reliable
        # is untouched pending the truth gates.
        if "island_stall_dchi2" in island_meta:
            out["island_stall_dchi2"] = island_meta["island_stall_dchi2"]
            _ptol = float((island_cfg or {}).get("plateau_dlnp_tol", 0.5))
            _pd = np.asarray(island_meta["island_stall_dchi2"])
            _conv_mode = str((island_cfg or {}).get(
                "convergence", "legacy")).strip().lower()
            if _conv_mode == "dlnp":
                # dlnp mode (E1a): certificate-passing reject-exits were
                # already mapped to CONVERGED in-stage (island_stalled
                # False there); island_plateau marks every reject-exit
                # whose ladder certificate passes -- finite stall_dchi2
                # is the reject-exit marker, stalled or not.
                out["island_plateau"] = (
                    np.isfinite(_pd) & (_pd > -2.0 * _ptol))
            else:
                out["island_plateau"] = (
                    np.asarray(island_meta["island_stalled"]).astype(bool)
                    & np.isfinite(_pd) & (_pd > -2.0 * _ptol))
        # Trust-region (step_mode="trust_region") bound-pinned disclosure:
        # converged fit pinned a free param at its Re-ceiling / n-cap /
        # position / ellipticity bound. The dogleg makes island_diverged's
        # clip-saturation signature vacuous, so this is its trust-region
        # analogue. Soft disclosure only -- never enters fit_is_reliable.
        # All-False in the default "clip" step mode.
        if "island_railed" in island_meta:
            out["island_railed"] = island_meta["island_railed"]
        # Item-4 Layer-2 blend_resolve (mode "resolve"): host rows whose
        # island was refit with an absorbed uncataloged star subtracted
        # (accepted by the chi2 not-worse guard). Disclosure column; the
        # star itself is a solve-only pass-2 sentinel column recorded in
        # {work_dir}/blend_resolved.csv, never a catalog row. Column
        # emitted only when the resolve pass ran and accepted something.
        if "blend_resolved" in island_meta and bool(
                np.any(island_meta["blend_resolved"])):
            out["blend_resolved"] = island_meta["blend_resolved"]
        # blend_unresolved: gate-passing candidate whose repair was
        # REJECTED -- positive uncorrectable-contamination evidence;
        # hard-fails fit_is_reliable by default
        # (merge.blend_unresolved_reliability).
        if "blend_unresolved" in island_meta and bool(
                np.any(island_meta["blend_unresolved"])):
            out["blend_unresolved"] = island_meta["blend_unresolved"]
            # append the pass-2 solved star fluxes to the ledger CSV
            if _sentinel_flux:
                try:
                    _brp = Path(cfg["outputs"]["work_dir"]) / \
                        "blend_resolved.csv"
                    if _brp.exists():
                        _brdf = pd.read_csv(_brp)
                        if "sent_id" in _brdf.columns:
                            for _bi4, _bn4 in enumerate(
                                    bandnames_with_psf):
                                _col = np.full(len(_brdf), np.nan)
                                for _q4, _sid4 in enumerate(
                                        _brdf["sent_id"].fillna(0)
                                        .astype(int)):
                                    _fv4 = _sentinel_flux.get(int(_sid4))
                                    if _fv4 is not None and _bi4 < len(_fv4):
                                        _col[_q4] = _fv4[_bi4]
                                _brdf[f"pass2_flux_{_bn4}"] = _col
                            _brdf.to_csv(_brp, index=False)
                except Exception as _e4:
                    logger.warning(
                        "blend_resolved.csv pass-2 flux update failed: "
                        "%s", _e4)
        if chi2_nu_enabled and "chi2_nu" in island_meta:
            # LEG D nonlinear diagnostic: per-island reduced chi2 (shared by
            # members; NaN off-island / giants). Applied to the nonlinear
            # *_err_tot columns below.
            out["chi2_nu_nl"] = island_meta["chi2_nu"]
        if emit_source_metrics:
            # #17 per-source model-quality columns (legacypipe DR9 convention).
            # PER-SOURCE (unlike chi2_nu_nl which is per-island): rchisq is the
            # own-profile-weighted reduced chi2, the fit-quality axis the
            # pipeline previously lacked. NaN off-island / giants.
            out["rchisq"] = island_meta["rchisq"]
            out["fracflux"] = island_meta["fracflux"]
            out["fracin"] = island_meta["fracin"]
            out["fracmasked"] = island_meta["fracmasked"]
            # #35 Option A disclosure: per-band data-constrained window
            # flux (island VarPro amplitude x unit-model window sum --
            # what the pixels actually constrained; FLUX_{band}_fit is
            # the untruncated-total extrapolation). Bands = the island
            # stage's PSF-carrying band list (meta self-describes the
            # order). NaN off-island / giants.
            fw = island_meta.get("flux_win")
            if fw is not None:
                for bi, bn in enumerate(
                        island_meta.get("flux_win_bands", [])):
                    out[f"FLUX_WIN_{bn}_fit"] = fw[:, bi]
        # Free-n bookkeeping comes from the island stage's own S/N gate
        # (the pass-2 tasks ran with a linear-only solver, which would
        # report every SERSIC row frozen). DEV rows keep a structurally
        # fixed n=4 (never freed) -> frozen too, so their reported
        # sersic_n_fit self-describes as a seed.
        out["sersic_n_frozen"] = (
            ((source_type == SOURCE_SERSIC) | (source_type == SOURCE_DEV))
            & ~island_meta["free_n"])
        # The island stage refits DEV/SERSIC shapes (the per-grid
        # per_source pass is overridden), so shape is NOT frozen for
        # island-fitted rows. Giant-island members freeze shape at the
        # seed, but that is disclosed by the island_giant column (and its
        # reliability wiring) rather than shape_frozen.
        out["shape_frozen"] = False
        if param_errors_enabled:
            # Island nonlinear-parameter 1-sigma errors (task #28a):
            # marginal over the island's kept nonlinear params + the
            # per-band fluxes (VarPro), background frozen; NaN where
            # undefined (stars' shapes, frozen n, giants, degenerate
            # Hessian). The meta arrays are full-catalog length with
            # NaN off-island, so whole-column assignment is exact.
            out["x_pix_white_err"] = island_meta["pos_x_err"]
            out["y_pix_white_err"] = island_meta["pos_y_err"]
            out["Re_err"] = island_meta["re_err"]
            out["ELL_err"] = island_meta["ell_err"]
            out["THETA_err"] = island_meta["theta_err"]
            # Fill the pre-created (NaN) sersic_n_err column for island
            # FREE-n rows. The per-grid fill above never fires in island
            # mode (pass-2 grids run per_source, whose sersic_n_err is
            # all-NaN), so this cannot collide with it.
            nerr_isl = np.asarray(island_meta["sersic_n_err"],
                                  dtype=np.float32)
            good_nerr = (np.isfinite(nerr_isl)
                         & np.asarray(island_meta["free_n"], dtype=bool))
            if np.any(good_nerr):
                out.loc[np.where(good_nerr)[0], "sersic_n_err"] = \
                    nerr_isl[good_nerr]
        if emit_prior_frac:
            # R1 prior_frac disclosure: fraction of each fitted param's
            # joint-fit constraint from the catalog prior (0 pure data ..
            # 1 pure prior; NaN off-island / no prior / giant). The pos/
            # shape/n errors above are now POSTERIOR (data+prior), so a
            # nonzero prior_frac flags a prior-informed value+error.
            out["x_pix_white_prior_frac"] = island_meta["pos_x_prior_frac"]
            out["y_pix_white_prior_frac"] = island_meta["pos_y_prior_frac"]
            out["Re_prior_frac"] = island_meta["re_prior_frac"]
            out["ELL_prior_frac"] = island_meta["ell_prior_frac"]
            out["THETA_prior_frac"] = island_meta["theta_prior_frac"]
            out["sersic_n_prior_frac"] = island_meta["sersic_n_prior_frac"]
            # (Per-band FLUX prior_frac is DEFERRED to the P7 field-level
            # flux solve; the reported FLUX/FLUXERR are the data-only
            # pass-2 S1' solve, so no flux prior applies to them.)
        logger.info(
            "grid_gpu[island]: island columns written -- %d fitted rows, "
            "unconverged %d, stalled %d (diverged %d), giant %d",
            int(fitm.sum()), int((fitm & ~conv).sum()),
            int(island_meta["island_stalled"].sum()),
            int(island_meta["island_diverged"].sum()),
            int(island_meta["island_giant"].sum()))

    # ---- Reporting error columns (FLUXERR_TOT, *_err_corr, *_err_tot, FLUXERR_MARG*, FLUX_MARG_FRAC):
    # built in fit/grid_gpu/error_columns.py (moved there unchanged by M2 unit 0.6).
    add_error_columns(
        out, cfg=cfg, bandnames=bandnames, bandnames_with_psf=bandnames_with_psf,
        error_model_cfg=error_model_cfg, confusion_enabled=confusion_enabled,
        correlated_noise_enabled=correlated_noise_enabled, fcorr_by_band=fcorr_by_band,
        chi2_nu_enabled=chi2_nu_enabled, island_meta=island_meta,
        param_errors_enabled=param_errors_enabled, emit_flux_marg=emit_flux_marg, log=logger)

    # flag_slow_converge (column-only soft quality warning; ISLAND mode only):
    # flag sources whose island did NOT converge within the solver's own fast
    # MAIN-PASS budget and was re-queued into the hard pass --
    #     opt_niters > compute.gpu.solver.island.max_steps_main
    # (fallback opt_niters >= max_steps when the re-queue is disabled). This uses
    # the solver's OWN "fast converger vs straggler" boundary: CONFIG-derived, no
    # truth-frozen iteration count (the old `slow_converge_niters: 12` was tuned
    # against injection truth and is removed), and robust to the field's actual
    # convergence distribution -- a spiky bulge field where >90% of islands
    # converge in 1 step breaks a median+MAD outlier test (NMAD==0 -> all-False),
    # but not the solver-budget line. Semantics of fit_is_reliable are NOT
    # changed. Restricted to position_solver=island because only there opt_niters
    # is a PER-SOURCE step count; the legacy per-grid solvers report a single
    # grid-level S3 count, so a per-source test is meaningless -- the column is
    # omitted for those solvers.
    slow_flag_on = error_model_cfg.get("slow_converge_flag", True)
    if island_meta is not None and slow_flag_on:
        niters_v = pd.to_numeric(
            out["opt_niters"], errors="coerce").fillna(0).to_numpy(dtype=float)
        has_fit = pd.to_numeric(
            out["x_pix_white_fit"], errors="coerce").notna().to_numpy()
        out["flag_slow_converge"] = _compute_slow_converge_flag(
            niters_v, has_fit,
            island_cfg.get("max_steps_main", 8),
            island_cfg.get("max_steps", 30))

    # flag_high_rchisq (#17, column-only unless the user opts in): per-source
    # own-profile-weighted reduced chi2 above the configured threshold. Emitted
    # only when the metric columns are present AND rchisq_flag_thresh is set
    # (default None -> no flag, columns only). merge.high_rchisq_reliability
    # ("warn" default) decides how it enters fit_is_reliable.
    rc_thr = error_model_cfg.get("rchisq_flag_thresh", None)
    if emit_source_metrics and rc_thr is not None and float(rc_thr) > 0:
        rc_v = pd.to_numeric(out["rchisq"], errors="coerce")
        has_fit_rc = pd.to_numeric(
            out["x_pix_white_fit"], errors="coerce").notna().to_numpy()
        out["flag_high_rchisq"] = (
            (rc_v.to_numpy() > float(rc_thr)) & has_fit_rc)

    # seed_shift_pix / flag_blend_suspect (item 4, matched-blend silent-
    # channel disclosure): the one measured contamination channel every
    # other flag misses is an uncataloged point source within ~1 PSF FWHM
    # of a cataloged source, whose white-stack blend detection lands inside
    # the bright-mask association radius of the host's own catalog entry --
    # "matched", so it escapes the bright mask, phantom routing AND
    # subtract_detected -- and the joint fit absorbs it: the host displaces
    # toward the interloper and over-brightens while fracflux / fracin /
    # FLUX_WIN / shape all stay reassuring (oracle: 64-95% of the star flux
    # absorbed, displacement exactly toward the star, fit_is_reliable
    # True). The two shipped observables that DO carry the signature are
    # the fit-vs-seed displacement and rchisq (oracle AUC 1.00 / 0.96);
    # their AND holds ~0.2% base rate on a crowded field.
    # seed_shift_pix is emitted unconditionally (pure geometry, no new
    # constants); the boolean flag is emitted under the SAME gate as
    # flag_high_rchisq (rchisq_flag_thresh set) and additionally requires
    # the displacement to exceed the run's own median PSF CORE sigma:
    # per band, FWHM_eff = 2*sqrt(A_halfmax/pi) from the central stamp's
    # half-maximum isophotal area (exact FWHM for a Gaussian; a pure
    # geometry of the run's own PSF, not tuned), sigma = FWHM_eff/2.355.
    # NOT the stamp second moment: r^2 weighting is wing-dominated for a
    # real (Moffat-like) PSF -- measured on the oracle field it gives
    # 4.66 px vs the 1.79 px core sigma and misses every silent blend
    # (displacements 3.1-4.3 px). Sub-core-sigma displacement is
    # compatible with noise / seed error; super-sigma displacement at
    # high rchisq is the blend signature (oracle: 3/3 silent hosts
    # caught at 0.20% reliable-row base rate). Soft disclosure only:
    # merge wires it as a fit_quality_issues tag, never flips
    # fit_is_reliable.
    fit_xv = pd.to_numeric(
        out["x_pix_white_fit"], errors="coerce").to_numpy(dtype=float)
    fit_yv = pd.to_numeric(
        out["y_pix_white_fit"], errors="coerce").to_numpy(dtype=float)
    seed_xv = pd.to_numeric(
        out["x_pix_white"], errors="coerce").to_numpy(dtype=float)
    seed_yv = pd.to_numeric(
        out["y_pix_white"], errors="coerce").to_numpy(dtype=float)
    seed_shift = np.hypot(fit_xv - seed_xv, fit_yv - seed_yv)
    out["seed_shift_pix"] = seed_shift
    psf_sigma_ref = float("nan")
    try:
        from .psf_polynomial import evaluate_psf_at_np as _eval_psf_np
        _sig_b = []
        for bn in bandnames_with_psf:
            _phi = np.clip(np.asarray(
                _eval_psf_np(psfs[bn], image_w * 0.5, image_h * 0.5),
                dtype=float), 0.0, None)
            _pk = float(_phi.max()) if _phi.size else 0.0
            if not np.isfinite(_pk) or _pk <= 0:
                continue
            _n_half = int(np.count_nonzero(_phi > 0.5 * _pk))
            if _n_half > 0:
                # FWHM_eff = 2*sqrt(A_halfmax/pi); sigma = FWHM/2.355
                _sig_b.append(
                    2.0 * float(np.sqrt(_n_half / np.pi)) / 2.355)
        if _sig_b:
            psf_sigma_ref = float(np.median(_sig_b))
    except Exception as e:
        logger.warning(
            "grid_gpu: PSF sigma for flag_blend_suspect unavailable "
            "(%s); flag skipped, seed_shift_pix still emitted",
            str(e)[:120])
    if (emit_source_metrics and rc_thr is not None and float(rc_thr) > 0
            and np.isfinite(psf_sigma_ref) and psf_sigma_ref > 0
            and "flag_high_rchisq" in out.columns):
        out["flag_blend_suspect"] = (
            out["flag_high_rchisq"].fillna(False).to_numpy(dtype=bool)
            & (seed_shift > psf_sigma_ref))
        logger.info(
            "grid_gpu: flag_blend_suspect -- %d source(s) flagged "
            "(rchisq > %.2f AND seed shift > median PSF sigma %.2f px)",
            int(out["flag_blend_suspect"].sum()), float(rc_thr),
            psf_sigma_ref)

    # R05: no-data band cells carry no value (all per-band flux columns NaN;
    # MAG follows). Last per-band column is built above; MAG is built below
    # or in compute_zp.
    _n_blank = blank_nodata_band_cells(out, bandnames)
    logger.info(
        "grid_gpu: no-data band cells blanked -- %d cell(s) that carried a "
        "finite FLUX without a flux error (per band: %s)",
        int(sum(_n_blank.values())),
        ", ".join(f"{b}={n}" for b, n in _n_blank.items() if n) or "none")

    # R06/R07: per-band reason flags + usable-band count. The PSF-lobe zone
    # is evaluated on the band's polynomial PSF at the fitted positions; a
    # band without a polynomial PSF gets no PSF bit, and says so.
    _psf_zone: dict = {}
    _no_zone: list = []
    _fx = pd.to_numeric(out["x_pix_white_fit"], errors="coerce").to_numpy(
        dtype=float)
    _fy = pd.to_numeric(out["y_pix_white_fit"], errors="coerce").to_numpy(
        dtype=float)
    _partial: dict = {}
    _t_partial = 0.0
    for bn in bandnames_with_psf:
        if getattr(psfs.get(bn), "coefs", None) is None:
            _no_zone.append(bn)
            continue
        _psf_zone[bn] = psf_lobe_zone(psfs[bn], _fx, _fy)
        # R30: part of the PSF on pixels without data in this band.
        _bits = (image_dict.get(bn) or {}).get("nodata_bits")
        if _bits is not None:
            _tp0 = time.perf_counter()
            _u = psf_uncovered_fraction(
                psfs[bn], np.unpackbits(_bits, axis=1, count=image_w).astype(bool),
                _fx, _fy)
            _partial[bn] = _u > PARTIAL_COVERAGE_MAX
            _t_partial += time.perf_counter() - _tp0
    # R11: per-band saturation hits from load_inputs / the Gaia injection
    # (internal "_sat_{band}" columns; phantom rows carry none -> False).
    from tract7dt.load_inputs import SATURATION_BAND_COLUMN_PREFIX as _SATP
    _sat_cols = [c for c in out.columns if c.startswith(_SATP)]
    _sat = {c[len(_SATP):]: out[c].fillna(False).astype(bool).to_numpy()
            for c in _sat_cols}
    _flag_n = compose_band_flags(out, bandnames, psf_zone=_psf_zone,
                                 saturated=_sat, partial=_partial)
    out.drop(columns=_sat_cols, inplace=True)
    logger.info(
        "grid_gpu: band flags -- PSF model with a negative lobe > %.1f%% "
        "of peak on measured cells per band: %s; saturated (value blanked) "
        "on measured cells per band: %s; > %.0f%% of the PSF on no-data "
        "pixels (partial coverage, value kept) on measured cells per band: "
        "%s [coverage check %.1f s]",
        100.0 * PSF_LOBE_FRAC,
        ", ".join(f"{b}={c['psf_lobe']}" for b, c in _flag_n.items()
                  if c["psf_lobe"]) or "none",
        ", ".join(f"{b}={c['saturated']}" for b, c in _flag_n.items()
                  if c["saturated"]) or "none",
        100.0 * PARTIAL_COVERAGE_MAX,
        ", ".join(f"{b}={c['partial_coverage']}" for b, c in _flag_n.items()
                  if c["partial_coverage"]) or "none",
        _t_partial)
    _no_cov = [b for b in bandnames_with_psf
               if b not in _partial and b not in _no_zone]
    if _no_cov:
        logger.warning(
            "grid_gpu: no partial-coverage check for band(s) %s (no no-data "
            "mask kept): their FLAGS carry no bit 16", ",".join(_no_cov))
    if _no_zone:
        logger.warning(
            "grid_gpu: no PSF-lobe check for band(s) %s (no polynomial "
            "PSF): their FLAGS carry no PSF bit", ",".join(_no_zone))

    # Delegate the actual fit_is_reliable / fit_quality_issues computation
    # to the SHARED helper in tract7dt.merge so GPU and CPU modes use
    # bit-identical semantics. Hard-fail conditions covered:
    #   excluded, no_fit_result, hit_max_iters, bound_stalled, bright_mask,
    #   island_stalled (island mode; hard-fail by default — see below).
    # Soft warnings (do not flip fit_is_reliable):
    #   not_converged, all_psf_fallback (latter only when PSF audit
    #   columns are present, which GPU mode does not currently produce).
    # merge.island_stalled_reliability selects how island_stalled enters the
    # contract: "fail" (default) hard-fails the flux-inflated stalled rows,
    # "warn" tags them softly, "ignore" restores pre-2026-07 behaviour.
    # merge.island_giant_reliability does the same for island_giant (giants
    # whose position+shape are the frozen catalog seed, "fail" by default).
    from tract7dt.merge import compute_reliability_flag
    merge_cfg = (cfg.get("merge", {}) or {})
    island_stalled_mode = str(
        merge_cfg.get("island_stalled_reliability", "fail")).strip().lower()
    island_giant_mode = str(
        merge_cfg.get("island_giant_reliability", "fail")).strip().lower()
    high_rchisq_mode = str(
        merge_cfg.get("high_rchisq_reliability", "warn")).strip().lower()
    blend_unresolved_mode = str(
        merge_cfg.get("blend_unresolved_reliability", "fail")).strip().lower()
    island_plateau_mode = str(
        merge_cfg.get("island_plateau_reliability", "stalled")).strip().lower()
    compute_reliability_flag(
        out, island_stalled_mode=island_stalled_mode,
        island_giant_mode=island_giant_mode,
        high_rchisq_mode=high_rchisq_mode,
        blend_unresolved_mode=blend_unresolved_mode,
        island_plateau_mode=island_plateau_mode)
    run_status.note("catalog", _catalog_summary(out, bandnames_with_psf))

    # ZP calibration pipeline reads MAG_*_fit; compute from FLUX_*_fit
    # using the same convention as the per-patch code:
    #   mag = -2.5 * log10(flux) + zp_ref      (zp_ref defaults to 25.0)
    # When ZP is enabled, compute_zp() drops these exact MAG_*/MAGERR_*_fit
    # columns and recomputes them with the calibrated per-band ZP, re-appending
    # them at the END of the frame. So writing the uncalibrated zp_ref versions
    # here is pure waste (~40 extra float columns through the 443 MB to_csv, then
    # re-read by compute_zp). Skip them in that case -- the final on-disk catalog
    # is byte-identical because compute_zp appends MAG/MAGERR at the end in band
    # order whether or not they were present first. Keep them when ZP is disabled
    # so the catalog still carries (uncalibrated) magnitudes.
    zp_enabled = bool(cfg.get("zp", {}).get("enabled", False))
    zp_ref = float(cfg.get("image_scaling", {}).get("zp_ref", 25.0))
    for bn in ([] if zp_enabled else bandnames):
        fluxcol = f"FLUX_{bn}_fit"
        magcol = f"MAG_{bn}_fit"
        magerrcol = f"MAGERR_{bn}_fit"
        fluxerrcol = f"FLUXERR_{bn}_fit"
        if fluxcol not in out.columns:
            continue
        f = out[fluxcol].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            mag = np.where(np.isfinite(f) & (f > 0),
                           -2.5 * np.log10(f) + zp_ref, np.nan)
        out[magcol] = mag
        if fluxerrcol in out.columns:
            fe = out[fluxerrcol].to_numpy(dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                magerr = np.where(np.isfinite(fe) & np.isfinite(f) & (f > 0),
                                  (2.5 / np.log(10.0)) * (fe / f), np.nan)
            out[magerrcol] = magerr


    # Exp C v2: phantom subtract-only rows are internal nuisance columns
    # of the S1 solves -- never emitted (user-catalog purity). Drop them
    # (and the marker column) as the LAST row operation before writing.
    if "_is_phantom_sub" in out.columns:
        _phm = out["_is_phantom_sub"].fillna(False).astype(bool)
        _nph = int(_phm.sum())
        out = out.loc[~_phm].drop(
            columns=["_is_phantom_sub"]).reset_index(drop=True)
        if _nph:
            logger.info("grid_gpu: dropped %d phantom subtract-only row(s) "
                        "from the output catalog", _nph)

    if zp_enabled:
        # In-memory handoff: compute_zp() will write the single final copy of
        # output_catalog.csv (it drops+recomputes MAG/MAGERR and rewrites the
        # whole catalog anyway). Skipping this ~443 MB write here and the
        # subsequent re-read in compute_zp avoids a redundant write+read of the
        # full catalog. The on-disk result is byte-identical (same in-memory
        # `out`, written once by compute_zp). state["gpu_output_df"] is consumed
        # by run_pipeline (_summarize_fit_results + compute_zp).
        state["gpu_output_df"] = out
        logger.info("grid_gpu: handing catalog to compute_zp in memory "
                    "(rows=%d, deferred write); built in %.2fs", len(out), time.perf_counter() - t0)
    else:
        out.to_csv(out_csv, index=False)
        logger.info("grid_gpu: wrote merged catalog %s in %.2fs (rows=%d)",
                    out_csv, time.perf_counter() - t0, len(out))

    logger.info("grid_gpu: DONE in %.1f s total wall",
                time.perf_counter() - t_total)
    return out_csv


__all__ = ["run_grid_gpu_fit"]
