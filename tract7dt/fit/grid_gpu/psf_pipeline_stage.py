"""Pipeline stage: build PSFEx-style direct-from-stars polynomial PSF
per band.

Runs after `bright_mask` and before `run_grid_gpu_fit` when
`compute.mode: gpu` AND `compute.gpu.psf.mode: from_stars` (default).
The dispatch decision lives in :func:`tract7dt.pipeline.run_pipeline`;
this module is unconditional once called.

Outputs are saved to ``{work_dir}/psf_polynomial/`` so the per-grid
worker can load them in the per-GPU subprocesses.
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .psf_from_stars import (
    PsfFromStarsResult, TrainingRows, build_psf_polynomial_from_stars_for_band,
    psf_training_rows,
)
from .shm_hygiene import start_parent_watchdog, sweep_stale_segments


logger = logging.getLogger("tract7dt.fit.grid_gpu.psf_pipeline_stage")

_DIAG_POOL: ProcessPoolExecutor | None = None
_DIAG_FUTURES: list[Future] = []


_BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def blas_threads_for(n_workers: int, *, logical_cpus: int | None = None,
                     threads_per_core: int | None = None) -> int:
    """BLAS threads per host worker = max(1, physical cores in this process's affinity // workers).

    D9-1 answer 3 (gpu-engineer): a pool of concurrent host workers must not each start a full
    OpenBLAS pool -- the default busy-waits, gives no wall-time gain and burns 17-36x the CPU
    (impl_a.1.md 6).  Measured at run time from the affinity and the core topology, no machine
    constant."""
    if logical_cpus is None:
        try:
            logical_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            logical_cpus = os.cpu_count() or 1
    if threads_per_core is None:
        try:
            import psutil  # type: ignore
            lg, ph = psutil.cpu_count(logical=True), psutil.cpu_count(logical=False)
            threads_per_core = max(1, int(lg) // int(ph)) if lg and ph else 1
        except ImportError:
            threads_per_core = 1
    physical = max(1, int(logical_cpus) // max(1, int(threads_per_core)))
    return max(1, physical // max(1, int(n_workers)))


@contextlib.contextmanager
def pinned_blas_env(n_threads: int):
    """Set OMP/OPENBLAS/MKL_NUM_THREADS for processes spawned inside the block (spawn workers
    read them when numpy loads, before any initializer runs), then restore the parent's values."""
    old = {k: os.environ.get(k) for k in _BLAS_ENV}
    for k in _BLAS_ENV:
        os.environ[k] = str(int(n_threads))
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _psf_worker_init() -> None:
    """ProcessPoolExecutor initializer for the PSF-from-stars workers.

    The default ``centering_method == "gw_centroid"`` path uses pure
    numpy + scipy.ndimage and DOES NOT import JAX, which used to cost
    ~5 s per spawn worker (~80 s of stage wall at 18 workers, observed
    in the pre-Apr-2026 run.log).

    The legacy ``centering_method == "epsfbuilder"`` path still imports
    JAX (for the per-star Lanczos shift), so we keep the JAX-friendly
    env-var defaults here as a no-op safety net for that case. We use
    ``setdefault`` so we don't clobber inherited values from the
    parent process.
    """
    # FIRST: exit when the parent dies, so a parent OOM kill can't strand
    # spawn workers blocked on their call queue (see shm_hygiene docs).
    start_parent_watchdog()

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # Same reason as the grid workers (development ledger SS23-25/32):
    # never let XLA spend mantissa bits it does not buy throughput with.
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    # Mirror the per-grid LM workers' allocator settings: even though
    # the gw_centroid path doesn't touch JAX, leaving these as the
    # platform allocator means *if* JAX does get imported (e.g. via
    # opt-in epsfbuilder centering) it won't preallocate a slab of GPU
    # memory in each PSF worker.
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _process_band_local(
    band: str,
    image: np.ndarray, sigma: np.ndarray, bad: np.ndarray, sat: np.ndarray,
    ps_cfg: dict[str, Any],
    rows: TrainingRows, linear_limit: float,
) -> tuple[str, "PsfFromStarsResult | None", str | None]:
    """Module-level worker (so ProcessPool can pickle it).

    Runs the full PSF-from-stars pipeline for one band and returns
    (band, result, error_string). Subprocess-friendly: all args are
    standard numpy/dict/astropy objects.
    """
    try:
        result = build_psf_polynomial_from_stars_for_band(
            band,
            np.asarray(image, dtype=np.float32),
            np.asarray(sigma, dtype=np.float32),
            np.asarray(bad, dtype=bool),
            sat_mask=np.asarray(sat, dtype=bool), cfg=ps_cfg,
            rows=rows, linear_limit=linear_limit,
        )
        return band, result, None
    except Exception as exc:
        import traceback
        return band, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _process_band_shm(
    band: str,
    img_shm_name: str, img_shape: tuple[int, int], img_dtype: str,
    sig_shm_name: str, sig_shape: tuple[int, int], sig_dtype: str,
    bad_shm_name: str, bad_shape: tuple[int, int], bad_dtype: str,
    sat_shm_name: str, sat_shape: tuple[int, int], sat_dtype: str,
    ps_cfg: dict[str, Any],
    out_dir_str: str | None = None, save_diag: bool = True,
    rows: TrainingRows | None = None, linear_limit: float = np.inf,
) -> tuple[str, dict | None, str | None]:
    """Shared-memory variant of :func:`_process_band_local`.

    Instead of pickling the per-band image arrays through the OS pipe
    (~480 MB per band x 18 bands = ~8.6 GB pickle traffic), the parent
    writes each band's arrays once into a POSIX shared-memory segment.
    Workers attach to the segment by name and view it as a numpy array
    via zero-copy ``np.ndarray(shape, dtype, buffer=shm.buf)``. Pickle
    cost drops from ~480 MB per band to a tiny tuple of (name, shape,
    dtype) strings.

    When ``out_dir_str`` is provided, the per-band IO + diagnostic
    montage is also done INSIDE THE WORKER so it parallelizes across
    bands instead of blocking the parent's as_completed loop. The
    parent only receives a small summary dict back; the heavy
    PsfFromStarsResult object is consumed in-process and freed.

    Without that (legacy path), the worker returns the full
    PsfFromStarsResult and the parent does the IO -- this serializes
    at the parent and was the dominant cost (~5 s/band x 18 bands)
    on the pre-Apr-2026 pipeline.
    """
    from multiprocessing import shared_memory
    try:
        shm_img = shared_memory.SharedMemory(name=img_shm_name)
        shm_sig = shared_memory.SharedMemory(name=sig_shm_name)
        shm_bad = shared_memory.SharedMemory(name=bad_shm_name)
        shm_sat = shared_memory.SharedMemory(name=sat_shm_name)
        try:
            img_view = np.ndarray(
                img_shape, dtype=np.dtype(img_dtype), buffer=shm_img.buf,
            )
            sig_view = np.ndarray(
                sig_shape, dtype=np.dtype(sig_dtype), buffer=shm_sig.buf,
            )
            bad_view = np.ndarray(
                bad_shape, dtype=np.dtype(bad_dtype), buffer=shm_bad.buf,
            )
            sat_view = np.ndarray(
                sat_shape, dtype=np.dtype(sat_dtype), buffer=shm_sat.buf,
            )
            if rows is None:
                raise ValueError("PSF training rows were not given")
            result = build_psf_polynomial_from_stars_for_band(
                band,
                np.array(img_view, dtype=np.float32, copy=True),
                np.array(sig_view, dtype=np.float32, copy=True),
                np.array(bad_view, dtype=bool, copy=True),
                sat_mask=np.array(sat_view, dtype=bool, copy=True), cfg=ps_cfg,
                rows=rows, linear_limit=linear_limit,
            )
            if out_dir_str is not None:
                # Worker-side IO: write the per-band npz/csv/png files
                # directly so the parent doesn't serialize on these
                # ~5 s/band of matplotlib + numpy savez_compressed work.
                out_dir = Path(out_dir_str)
                qc = _save_one_psf(out_dir, band, result, save_diag)
                summary = {
                    **qc,
                    "n_used": int(result.n_stars_used_in_fit),
                    "n_after_gaia": int(result.n_stars_after_gaia),
                    "n_after_quality": int(result.n_stars_after_quality),
                    "fwhm_mode": float(result.fwhm_mode),
                    "outer_iters": int(result.n_outer_iters),
                    "timing_sec": float(result.timing_sec),
                }
                return band, summary, None
            # Legacy path: return full result; parent handles IO.
            return band, result, None  # type: ignore[return-value]
        finally:
            shm_img.close()
            shm_sig.close()
            shm_bad.close()
            shm_sat.close()
    except Exception as exc:
        import traceback
        return band, None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _save_psf_residual_montage(
    out_path: Path, band: str, result: "PsfFromStarsResult",
    n_show: int = 25,
) -> None:
    """Per-band PSF model-vs-actual diagnostic plot.

    Selects up to ``n_show`` stars (~5x5 grid) spread across the image,
    and for each draws a 3-panel row: ACTUAL stamp / MODEL stamp at
    star's (x, y) / SIGNED RESIDUAL (actual - model). The model is the
    polynomial-PSF evaluated at the star's exact fitted position so the
    residual measures TRUE polynomial fit error (not centering error).

    Layout: ``n_show`` rows x 3 columns.

    Uses the numpy-only :func:`evaluate_psf_at_batch_np` so this
    diagnostic-only function does not import JAX in the pipeline parent
    process. (The previous JAX path imported jax in the parent on the
    first per-band callback, blocking the as_completed loop and
    serializing what should have been parallel band completions.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from .psf_polynomial import evaluate_psf_at_batch_np

    psf = result.psf_polynomial
    n_stars = result.n_stars_used_in_fit
    if n_stars == 0:
        return

    # Spatially-stratified pick: n_show stars distributed across the
    # image (sort by x, then take every k-th).
    order = np.argsort(result.star_x)
    if n_stars > n_show:
        step = max(1, n_stars // n_show)
        pick = order[::step][:n_show]
    else:
        pick = order
    n_pick = len(pick)

    actual_stamps = result.star_centered_stamps[pick]  # (n_pick, H, W)
    star_x = result.star_x[pick]
    star_y = result.star_y[pick]
    model_stamps = evaluate_psf_at_batch_np(psf, star_x, star_y)
    # Amplitude-fit each model to its star (see the from-files montage): the
    # residual then shows true PSF-shape error, not per-star flux scatter.
    _mm = (model_stamps * model_stamps).sum(axis=(1, 2))
    alpha = np.where(_mm > 0, (actual_stamps * model_stamps).sum(axis=(1, 2)) / _mm, 1.0)
    model_scaled = alpha[:, None, None] * model_stamps
    residuals = actual_stamps - model_scaled
    RESID_FRAC = 0.05  # fixed residual color scale = +/-5% of each star's peak

    fig, axes = plt.subplots(
        n_pick, 3, figsize=(7.5, 2.0 * n_pick + 0.5), squeeze=False,
    )
    fig.suptitle(
        f"{band}: PSF model vs actual (flux-fit; resid scale +/-{RESID_FRAC*100:.0f}% of peak; "
        f"order={result.polynomial_order}, {result.n_stars_used_in_fit} stars)",
        fontsize=11, y=1.0 - 0.5 / (n_pick + 1),
    )

    for i in range(n_pick):
        a = actual_stamps[i]
        ms = model_scaled[i]
        d = residuals[i]
        x_pix, y_pix = float(star_x[i]), float(star_y[i])
        peak_a = float(max(a.max(), 1e-30))
        vmin = float(np.percentile(np.concatenate([a.ravel(), ms.ravel()]), 2))
        vmax = float(np.percentile(np.concatenate([a.ravel(), ms.ravel()]), 99.5))
        if vmax <= vmin:
            vmax = vmin + 1e-30
        rs = RESID_FRAC * peak_a   # FIXED scale (not auto-stretched)
        norm_d = mcolors.TwoSlopeNorm(vmin=-rs, vcenter=0.0, vmax=rs)
        max_pct = float(np.max(np.abs(d)) / peak_a) * 100

        for j, (img, cmap, norm, vmin_l, vmax_l, title) in enumerate([
            (a, "viridis", None, vmin, vmax,
             f"actual\npeak={a.max():.4f}"),
            (ms, "viridis", None, vmin, vmax,
             f"flux*model @ ({x_pix:.0f},{y_pix:.0f})\nflux ratio={alpha[i]:.3f}"),
            (d, "RdBu_r", norm_d, None, None,
             f"actual - flux*model\nmax%={max_pct:.1f}, rms={float(np.sqrt((d * d).mean())):.2e}"),
        ]):
            ax = axes[i][j]
            if norm is not None:
                im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
            else:
                im = ax.imshow(img, cmap=cmap, vmin=vmin_l, vmax=vmax_l,
                               interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=6)

    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.6 / (n_pick + 1)))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_psf_residual_montage_from_files(
    band_dir: str | Path,
    band: str,
    n_show: int = 25,
) -> tuple[str, str | None]:
    """Build the residual montage from already-written diagnostic files."""
    try:
        band_dir = Path(band_dir)
        from .psf_polynomial import evaluate_psf_at_batch_np
        psf = load_psf_polynomial_for_band(band_dir.parent, band)
        with np.load(band_dir / "stars_centered_stamps.npz") as data:
            star_centered_stamps = data["stamps"].astype(np.float32, copy=False)
            star_x = data["star_x"].astype(np.float64, copy=False)
            star_y = data["star_y"].astype(np.float64, copy=False)
        try:
            with open(band_dir / "summary.json", "r") as f:
                meta = json.load(f)
            polynomial_order = int(meta.get("polynomial_order", psf.coefs.shape[-1]))
            n_stars_used = int(meta.get("n_stars_used_in_fit", len(star_x)))
        except Exception:
            polynomial_order = int(psf.coefs.shape[-1])
            n_stars_used = int(len(star_x))

        n_stars = int(len(star_x))
        if n_stars == 0:
            return band, None
        order = np.argsort(star_x)
        if n_stars > n_show:
            step = max(1, n_stars // n_show)
            pick = order[::step][:n_show]
        else:
            pick = order
        n_pick = len(pick)
        actual_stamps = star_centered_stamps[pick]
        pick_x = star_x[pick]
        pick_y = star_y[pick]
        model_stamps = evaluate_psf_at_batch_np(psf, pick_x, pick_y)
        # Amplitude-fit each model to its star (flux is a free parameter in
        # forced photometry), so the residual shows the TRUE PSF-SHAPE error
        # rather than per-star flux scatter. Without this, a unit-sum
        # comparison renders ~13% "central excess" that is purely the +/-flux
        # scatter the flux fit absorbs (the shape residual is ~1-2%).
        _mm = (model_stamps * model_stamps).sum(axis=(1, 2))
        alpha = np.where(_mm > 0, (actual_stamps * model_stamps).sum(axis=(1, 2)) / _mm, 1.0)
        model_scaled = alpha[:, None, None] * model_stamps
        residuals = actual_stamps - model_scaled
        RESID_FRAC = 0.05  # fixed residual color scale = +/-5% of each star's peak

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        fig, axes = plt.subplots(
            n_pick, 3, figsize=(7.5, 2.0 * n_pick + 0.5), squeeze=False,
        )
        fig.suptitle(
            f"{band}: PSF model vs actual (flux-fit; resid scale +/-{RESID_FRAC*100:.0f}% of peak; "
            f"order={polynomial_order}, {n_stars_used} stars)",
            fontsize=11, y=1.0 - 0.5 / (n_pick + 1),
        )
        for i in range(n_pick):
            a = actual_stamps[i]
            ms = model_scaled[i]
            d = residuals[i]
            x_pix, y_pix = float(pick_x[i]), float(pick_y[i])
            peak_a = float(max(a.max(), 1e-30))
            vmin = float(np.percentile(np.concatenate([a.ravel(), ms.ravel()]), 2))
            vmax = float(np.percentile(np.concatenate([a.ravel(), ms.ravel()]), 99.5))
            if vmax <= vmin:
                vmax = vmin + 1e-30
            rs = RESID_FRAC * peak_a   # FIXED scale (not auto-stretched)
            norm_d = mcolors.TwoSlopeNorm(vmin=-rs, vcenter=0.0, vmax=rs)
            max_pct = float(np.max(np.abs(d)) / peak_a) * 100

            for j, (img, cmap, norm, vmin_l, vmax_l, title) in enumerate([
                (a, "viridis", None, vmin, vmax,
                 f"actual\npeak={a.max():.4f}"),
                (ms, "viridis", None, vmin, vmax,
                 f"flux*model @ ({x_pix:.0f},{y_pix:.0f})\nflux ratio={alpha[i]:.3f}"),
                (d, "RdBu_r", norm_d, None, None,
                 f"actual - flux*model\nmax%={max_pct:.1f}, rms={float(np.sqrt((d * d).mean())):.2e}"),
            ]):
                ax = axes[i][j]
                if norm is not None:
                    im = ax.imshow(img, cmap=cmap, norm=norm, interpolation="nearest")
                else:
                    im = ax.imshow(img, cmap=cmap, vmin=vmin_l, vmax=vmax_l,
                                   interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(title, fontsize=8)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cbar.ax.tick_params(labelsize=6)

        fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.6 / (n_pick + 1)))
        fig.savefig(band_dir / "psf_residual.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return band, None
    except Exception as exc:
        import traceback
        return band, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _diag_worker_init() -> None:
    """Initializer for the montage writers: the parent watchdog, plus the
    lowest scheduling priority.

    effarc2 K10 (2026-08-26): these are WRITE-ONLY diagnostic PNGs, but they
    are matplotlib renders of 25 stars x 3 panels per band and they run
    concurrently with the phases that decide the run's wall time. Profiling
    the sparse anchor put `_save_psf_residual_montage_from_files` at ~26% of
    all worker-thread samples -- four cores of matplotlib competing with the
    XLA compiles and the host FFT passes on a box that is already
    oversubscribed. `nice(19)` costs the diagnostics nothing (they only have
    to be finished by the end of the run, and the drain already waits) and
    stops them from taking CPU away from the critical path. Not a tuning
    constant: 19 is "run only on otherwise-idle CPU", the single meaningful
    value for a write-only diagnostic.
    """
    start_parent_watchdog()
    try:
        os.nice(19)
    except Exception:  # noqa: BLE001  (never let a nicety break a run)
        pass


def _submit_psf_residual_montage(out_dir: Path, band: str) -> None:
    """Queue one residual montage write in a background process pool."""
    global _DIAG_POOL, _DIAG_FUTURES
    if _DIAG_POOL is None:
        ctx = mp.get_context("spawn")
        _DIAG_POOL = ProcessPoolExecutor(
            max_workers=min(4, max(1, os.cpu_count() or 1)),
            mp_context=ctx,
            # Watchdog so a parent OOM kill can't strand these background
            # writers as immortal orphans pinning the resource_tracker.
            initializer=_diag_worker_init,
        )
    fut = _DIAG_POOL.submit(
        _save_psf_residual_montage_from_files,
        str(out_dir / band),
        band,
        25,
    )
    _DIAG_FUTURES.append(fut)


def shutdown_psf_diagnostics() -> None:
    """Drain background PSF diagnostic writers, if any.

    Bounded wait: a wedged montage worker must not hang the whole
    pipeline at exit (the drain runs in run_pipeline's finally). Montages
    take seconds; anything past the timeout is abandoned and its worker
    killed so the executor shutdown cannot block on it either.
    """
    global _DIAG_POOL, _DIAG_FUTURES
    if _DIAG_POOL is None:
        return
    pending = list(_DIAG_FUTURES)
    n_failed = 0
    stuck = False
    for fut in pending:
        try:
            band, err = fut.result(timeout=300)
            if err:
                n_failed += 1
                logger.warning("psf_from_stars: residual montage failed for %s: %s", band, err)
        except TimeoutError:
            n_failed += 1
            stuck = True
            logger.warning(
                "psf_from_stars: residual montage worker timed out (>300s); abandoning"
            )
        except Exception as exc:
            n_failed += 1
            logger.warning("psf_from_stars: residual montage worker failed: %s", exc)
    if stuck:
        for p in list(getattr(_DIAG_POOL, "_processes", {}).values()):
            try:
                p.kill()
            except Exception:  # noqa: BLE001 - best-effort at shutdown
                pass
    _DIAG_POOL.shutdown(wait=True, cancel_futures=stuck)
    _DIAG_POOL = None
    _DIAG_FUTURES = []
    if pending:
        logger.info(
            "psf_from_stars: background residual montages drained: %d saved, %d failed",
            len(pending) - n_failed, n_failed,
        )


def _log_integral_qc(band: str, qc: dict) -> None:
    """Parent-side (run.log) warning for a spatially non-flat stamp integral."""
    lo = float(qc.get("integral_p10", float("nan")))
    hi = float(qc.get("integral_p90", float("nan")))
    if np.isfinite(lo) and np.isfinite(hi) and hi / max(lo, 1e-6) > 1.05:
        logger.warning(
            "psf_from_stars[%s]: PSF stamp INTEGRAL is not spatially flat "
            "(p10 %.3f / p90 %.3f over a 5x5 grid). Reported STAR fluxes of "
            "all-STAR grids are renormalised per position; b_hat, the "
            "confusion error, scene subtraction, island fits and grids with "
            "galaxies use the raw stamp and carry the local integral.",
            band, lo, hi)


def _save_one_psf(
    out_dir: Path, band: str, result: PsfFromStarsResult,
    save_diagnostics: bool,
) -> dict:
    """Persist the polynomial PSF for one band to disk.

    Returns the stamp-integral QC (``integral_p10`` / ``integral_p90``, NaN
    when it could not be evaluated) for the PARENT to log: this function runs
    in a spawn worker whose log lines reach only the launcher's stderr, never
    run.log (R07).
    """
    band_dir = out_dir / band
    band_dir.mkdir(parents=True, exist_ok=True)

    # Polynomial PSF: store coefs + normalization + metadata as a single npz.
    # star_x/star_y = the training stars used in the fit (their convex hull
    # is where the spatial polynomial interpolates, the rest is
    # extrapolation). Always saved, unlike the diagnostic stars.csv.
    np.savez(
        band_dir / "psf_polynomial.npz",
        coefs=result.psf_polynomial.coefs.astype(np.float32),
        x_origin=np.float32(result.psf_polynomial.x_origin),
        x_scale=np.float32(result.psf_polynomial.x_scale),
        y_origin=np.float32(result.psf_polynomial.y_origin),
        y_scale=np.float32(result.psf_polynomial.y_scale),
        H_psf=np.int32(result.psf_polynomial.H_psf),
        W_psf=np.int32(result.psf_polynomial.W_psf),
        polynomial_order=np.int32(result.polynomial_order),
        star_x=np.asarray(result.star_x, dtype=np.float64),
        star_y=np.asarray(result.star_y, dtype=np.float64),
    )
    # QC disclosure: the stamp INTEGRAL should be spatially flat. The
    # median wing surgery (_replace_model_wings_with_median) breaks the unit
    # integral away from the field centre (T08676 0.84-1.03, T08285
    # 0.90-1.14 at source positions; verify/VB-psf.md VB1). Reported STAR
    # fluxes of all-STAR grids are renormalised per position
    # (forced_phot_sparse.render_stamps), but b_hat, the confusion error,
    # scene subtraction, island LM and grids holding galaxies use the raw
    # stamp, so they carry the local integral. The parent warns when the
    # 10-90% spread exceeds 5% (policy disclosure threshold). Repair = 3a.
    qc = {"integral_p10": float("nan"), "integral_p90": float("nan")}
    try:
        from .psf_polynomial import evaluate_psf_at_np
        _xg = result.psf_polynomial.x_origin + result.psf_polynomial.x_scale \
            * np.linspace(-1.0, 1.0, 5)
        _yg = result.psf_polynomial.y_origin + result.psf_polynomial.y_scale \
            * np.linspace(-1.0, 1.0, 5)
        _ss = np.array([
            float(np.sum(evaluate_psf_at_np(result.psf_polynomial, _x, _y)))
            for _y in _yg for _x in _xg])
        qc["integral_p10"] = float(np.percentile(_ss, 10))
        qc["integral_p90"] = float(np.percentile(_ss, 90))
    except Exception as _e:  # noqa: BLE001 - QC must never break the save
        logger.debug("psf integral QC skipped (%s)", _e)

    # Diagnostics
    if save_diagnostics:
        meta = {
            "band": band,
            "polynomial_order": int(result.polynomial_order),
            "n_stars_detected": int(result.n_stars_detected),
            "n_stars_after_quality": int(result.n_stars_after_quality),
            "n_stars_after_gaia": int(result.n_stars_after_gaia),
            "n_stars_used_in_fit": int(result.n_stars_used_in_fit),
            "fwhm_mode_pix": float(result.fwhm_mode),
            "fwhm_min_pix": float(result.fwhm_min),
            "fwhm_max_pix": float(result.fwhm_max),
            "outer_iters": int(result.n_outer_iters),
            "final_chi2_mean": float(result.final_chi2_mean),
            "final_chi2_max": float(result.final_chi2_max),
            "timing_sec": float(result.timing_sec),
            "psf_size": [
                int(result.psf_polynomial.H_psf),
                int(result.psf_polynomial.W_psf),
            ],
            "coef_shape": list(result.psf_polynomial.coefs.shape),
            # --- crowding-robust path diagnostics (success metrics) ---
            "psf_normalization_used": str(result.psf_normalization_used),
            "r_core_pix": float(result.r_core_pix),
            "median_neighbor_pix_frac": float(result.median_neighbor_pix_frac),
            "central_resid_frac": float(result.central_resid_frac),
            "central_monopole_frac": float(result.central_monopole_frac),
            "central_dipole_frac": float(result.central_dipole_frac),
            "n_singular_px": int(result.n_singular_px),
            "min_neff_px": float(result.min_neff_px),
        }
        with open(band_dir / "summary.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Per-star catalog (used in fit) for downstream diagnostics
        # (e.g. PSF residual visualization). Extra crowding columns are
        # included when the crowding-robust path populated them (they are
        # empty arrays on the legacy path -> column omitted to stay aligned).
        cols = {
            "star_x": result.star_x.astype(np.float64),
            "star_y": result.star_y.astype(np.float64),
            "star_chi2": result.star_chi2.astype(np.float64),
            "star_flux": result.star_flux.astype(np.float64),
            "star_fwhm_pix": result.star_fwhm.astype(np.float64),
            "star_ellipticity": result.star_ellip.astype(np.float64),
            "star_origin": result.star_origin.astype(str),
        }
        n_used = len(result.star_x)
        for name, arr in (
            ("star_local_bkg", result.star_local_bkg),
            ("star_neighbor_pix_frac", result.star_neighbor_pix_frac),
            ("star_masked_pix_frac", result.star_masked_pix_frac),
            ("star_core_norm_flux", result.star_core_norm_flux),
            ("star_weighted_chi2", result.star_weighted_chi2),
            ("star_n_neighbors", result.star_n_neighbors),
        ):
            if arr is not None and len(arr) == n_used:
                cols[name] = np.asarray(arr)
        df = pd.DataFrame(cols)
        df.to_csv(band_dir / "stars.csv", index=False)
        # Per-star centered cutouts (small array; < 100 MB even for a few
        # thousand stars at 61x61 f32). Uncompressed savez: this is on the gating
        # band's critical path and is read back via np.load (which transparently
        # handles uncompressed) for the async residual montage -- identical arrays,
        # faster write+read, only the on-disk encoding differs.
        np.savez(
            band_dir / "stars_centered_stamps.npz",
            stamps=result.star_centered_stamps.astype(np.float32),
            star_x=result.star_x.astype(np.float64),
            star_y=result.star_y.astype(np.float64),
        )
    return qc


def _band_linear_limit(band_data: dict) -> tuple[float, str]:
    """Linear limit of one band in image units for the PSF training stars (design 1.5).

    The ``saturation.linear_limit`` setting of the input contract (unit 0.5) does not exist
    yet, so this is its stated fallback: the header SATURATE (raw ADU) times the band's
    flux scale. No finite SATURATE: no limit (inf; the saturation mask is empty too).
    """
    hdr = band_data.get("hdr")
    raw = hdr.get("SATURATE") if hdr is not None else None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        v = float("nan")
    if np.isfinite(v) and v > 0:
        return v * float(band_data["scale"]), (
            f"header SATURATE {v:g} ADU (no saturation.linear_limit setting)")
    return float("inf"), "none (no finite header SATURATE)"


def _band_saturation_mask(band_data: dict) -> np.ndarray:
    """The band's saturated-pixel mask (load_inputs: pixel >= SATURATE / saturation_divisor
    in the original frame) -- not the catalogue column ``_sat_{band}`` and not the bright
    mask (which covers the unsaturated bright stars that train the PSF)."""
    sat = band_data.get("satur_mask")
    if sat is None:
        return np.zeros(np.shape(band_data["img_scaled"]), dtype=bool)
    return np.asarray(sat, dtype=bool)


def _training_inputs(cfg: dict, state: dict, bandnames) -> dict:
    """Per band: (TrainingRows, linear limit in image units, its source).

    The rows depend on the band's WCS only, so bands that share a WCS share one mapping.
    """
    cat = state["input_catalog"]
    max_ruwe = float(cfg["zp"]["gaia_max_ruwe"])
    by_wcs: dict[str, TrainingRows] = {}
    out = {}
    for bn in bandnames:
        bd = state["image_dict"][bn]
        wcs = bd["wcs"]
        key = wcs.to_header_string(relax=True)
        if key not in by_wcs:
            by_wcs[key] = psf_training_rows(
                cat, lambda ra, dec, _w=wcs: _w.all_world2pix(ra, dec, 0, quiet=True),
                max_ruwe=max_ruwe)
            rows = by_wcs[key]
            (logger.warning if rows.origin == "star" else logger.info)(
                "psf_from_stars: %s (band %s%s)", rows.note, bn,
                "" if len(bandnames) == 1 else " and every band with the same WCS")
        limit, src = _band_linear_limit(bd)
        (logger.info if np.isfinite(limit) else logger.warning)(
            "psf_from_stars: [%s] linear limit of PSF training stars: %s%s", bn, src,
            "" if np.isfinite(limit) else " -- no saturation exclusion in this band (the saturation "
            "mask is empty too)")
        out[bn] = (by_wcs[key], limit, src)
    return out


def _rows_digest(rows: TrainingRows) -> str:
    import hashlib
    h = hashlib.sha256()
    for a in (rows.x, rows.y, rows.trainable):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(rows.origin.encode())
    return h.hexdigest()


def _psf_cache_key(cfg: dict, ps_cfg: dict, band_data: dict,
                   training: tuple | None = None) -> dict:
    """What a cached PSF of one band must have been built from."""
    fp = Path(str(band_data.get("fp", "")))
    try:
        st = fp.stat()
        size, mtime_ns = int(st.st_size), int(st.st_mtime_ns)
    except OSError:
        size, mtime_ns = -1, -1
    key = {
        "frame": str(fp), "frame_size": size, "frame_mtime_ns": mtime_ns,
        "shape": [int(v) for v in np.shape(band_data.get("img_scaled", []))],
        "from_stars": ps_cfg,
        "image_scaling": cfg.get("image_scaling", {}),
        "crop": cfg.get("crop", {}),
        "saturation_divisor": (cfg.get("source_saturation_cut", {}) or {})
        .get("saturation_divisor"),
    }
    if training is not None:
        rows, limit, _ = training
        key["training_rows_sha256"] = _rows_digest(rows)
        key["linear_limit"] = None if not np.isfinite(limit) else float(limit)
    # JSON round trip: the comparison is against a key read back from disk
    return json.loads(json.dumps(key, sort_keys=True, default=str))


def build_psf_polynomial_from_stars_stage(
    cfg: dict[str, Any], state: dict[str, Any],
) -> Path:
    """Pipeline stage: build the per-band polynomial PSF model from
    individual SEP-detected stars (PSFEx-style).

    Saves results to ``{work_dir}/psf_polynomial/<band>/`` and
    populates ``state['psf_polynomial_dir']``.

    The caller (run_pipeline) decides whether to invoke this stage
    based on compute.gpu.psf.mode; this function is unconditional.
    """
    gpu_cfg = cfg.get("compute", {}).get("gpu", {}) or {}
    psf_cfg_block = gpu_cfg.get("psf", {}) or {}
    ps_cfg = psf_cfg_block.get("from_stars", {}) or {}

    image_dict = state["image_dict"]
    bandnames = sorted(image_dict.keys())
    n_bands = len(bandnames)

    # PSF training stars (M2 unit a.3, design 1.5): SEP detections paired with the input
    # rows that have a Gaia identity and pass the Gaia quality (or, in a list without Gaia
    # identity, the TYPE == STAR rows), minus the data properties in select_psf_stars.
    training = _training_inputs(cfg, state, bandnames)

    out_dir = Path(cfg["outputs"]["work_dir"]) / "psf_polynomial"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CACHE CHECK: if every band already has a psf_polynomial.npz whose
    # cache_key.json matches this run's key, skip the entire build stage
    # (parameter sweeps that vary (bg_order, tol) but share the PSF inputs
    # reuse PSFs, ~150-300 s of PSF stage wall). The key is the band's
    # source frame (path, size, mtime) + every config block that shapes the
    # PSF input or build; the old key (polynomial order only) reused a PSF
    # built from another frame of the same band.
    from .bg_polynomial import n_terms_for_order
    expected_order = int(ps_cfg.get("polynomial_order", 2))
    expected_n_terms = n_terms_for_order(expected_order)
    cache_keys = {bn: _psf_cache_key(cfg, ps_cfg, image_dict[bn], training[bn])
                  for bn in bandnames}
    cached_bands = []
    for bn in bandnames:
        npz_path = out_dir / bn / "psf_polynomial.npz"
        key_path = out_dir / bn / "cache_key.json"
        if not (npz_path.exists() and key_path.exists()):
            continue
        try:
            with open(key_path) as f:
                stored_key = json.load(f)
            with np.load(npz_path) as data:
                # Stored shape: (H_psf, W_psf, n_terms)
                n_terms_stored = int(data["coefs"].shape[-1])
            if (stored_key == cache_keys[bn]
                    and n_terms_stored == expected_n_terms):
                cached_bands.append(bn)
        except Exception as exc:
            logger.warning(
                "psf_from_stars: cache check failed for %s: %s", bn, exc,
            )
    if len(cached_bands) == n_bands:
        logger.info(
            "psf_from_stars: ALL %d bands already have cached "
            "psf_polynomial.npz at order %d in %s -> SKIPPING PSF build",
            n_bands, expected_order, out_dir,
        )
        state["psf_polynomial_dir"] = str(out_dir)
        return out_dir

    # Saturation: each band's saturated-pixel mask (image_dict[band]["satur_mask"], see
    # _band_saturation_mask) and its linear limit go to select_psf_stars. The bright mask is
    # not used here: it covers every bright source, including the unsaturated bright stars
    # that train the PSF (and it is only folded into 'bad' after this stage).

    # Per-band parallelism via ProcessPoolExecutor. We tried
    # ThreadPoolExecutor first but photutils' EPSFBuilder is GIL-bound
    # (Python loops over star objects), so threads serialize and add
    # context-switch overhead. ProcessPool gives true parallelism, at
    # the cost of pickling the per-band image arrays (~700 MB) + per-
    # process Python startup (~3 s on cold cache). For per-band work
    # of ~30-50 s, the IPC + startup overhead is well-amortized.
    save_diag = bool(ps_cfg.get("save_diagnostics", True))
    max_workers_cfg = ps_cfg.get("max_workers", 4)
    mw_str = str(max_workers_cfg).strip().lower()
    if mw_str == "all_bands":
        max_workers = n_bands
    elif mw_str == "auto":
        n_cpus = os.cpu_count() or 4
        max_workers = min(n_bands, max(1, n_cpus // 8))
    else:
        max_workers = max(1, int(max_workers_cfg))
    if max_workers > n_bands:
        logger.info(
            "psf_from_stars: max_workers=%d > n_bands=%d -> capping to %d",
            max_workers, n_bands, n_bands,
        )
        max_workers = n_bands

    # Memory-aware soft cap: each subprocess peaks at ~1.3 GB on the
    # crowding-robust path (image f32 + int32 segmap + per-pixel weight/
    # residual f64 matrices). Cap so we don't risk swap thrashing.
    try:
        import psutil  # type: ignore
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        # Reserve ~5 GB headroom for OS + the eventual grid_gpu fit step.
        usable_for_psf_gb = max(0.5, avail_gb - 5.0)
        proc_cost_gb = 1.3
        memcap = max(1, int(usable_for_psf_gb / proc_cost_gb))
        if memcap < max_workers:
            logger.warning(
                "psf_from_stars: max_workers=%d would need ~%.1f GB but only "
                "%.1f GB free; capping to %d to avoid swap thrashing",
                max_workers, max_workers * proc_cost_gb, avail_gb, memcap,
            )
            max_workers = memcap
    except ImportError:
        pass
    logger.info("psf_from_stars: running %d bands w/ %d worker process(es)",
                n_bands, max_workers)
    t_start = time.perf_counter()
    summary_rows: list[dict] = []

    if max_workers == 1:
        # In-process serial: faster for single-band (no pickle/IPC).
        results_list: list[tuple] = []
        for bn in bandnames:
            band_data = image_dict[bn]
            rows_b, limit_b, _ = training[bn]
            results_list.append(_process_band_local(
                bn, band_data["img_scaled"], band_data["sigma_total_scaled"],
                band_data["bad"], _band_saturation_mask(band_data), ps_cfg,
                rows_b, limit_b,
            ))
    else:
        # ProcessPool with SHARED MEMORY: each band's arrays
        # (img_scaled + sigma_total_scaled + bad + saturation mask, ~550 MB) is
        # written ONCE into a POSIX shared-memory segment. Workers
        # attach by name (a few bytes pickled) instead of receiving
        # ~480 MB through the OS pipe. Saves ~17 s of effective
        # serialized pickle traffic (8.6 GB at ~500 MB/s pipe limit)
        # at the cost of one big upfront copy into shm (~3-5 s).
        # Use 'spawn' to avoid forking the parent's CUDA context.
        from multiprocessing import shared_memory
        ctx = mp.get_context("spawn")

        # Sweep stale segments from previous crashed runs before allocating
        # (a kernel OOM SIGKILL bypasses the cleanup `finally` below, and
        # the leaked tmpfs pages are RAM stolen from this run).
        sweep_stale_segments(logger)

        # Allocate shared-memory segments for all bands UPFRONT and
        # copy the per-band arrays into them. Keep the SharedMemory
        # handles in scope until all workers finish (otherwise they'd
        # be GC'd and unlinked).
        shm_handles: list[shared_memory.SharedMemory] = []
        tasks = []
        _pid = os.getpid()

        def _alloc_band_shm(bn):
            # Allocate the 4 shm segments for one band and copy its arrays in.
            # NOTE: out_dir + save_diag travel in the task tuple so the worker
            # writes its own per-band npz/csv/png (parallelizing the ~5 s/band of
            # matplotlib + savez work that used to serialize in the parent).
            bd = image_dict[bn]
            arrs = [
                ("img", np.ascontiguousarray(bd["img_scaled"], dtype=np.float32)),
                ("sig", np.ascontiguousarray(bd["sigma_total_scaled"], dtype=np.float32)),
                ("bad", np.ascontiguousarray(bd["bad"], dtype=bool)),
                ("sat", np.ascontiguousarray(_band_saturation_mask(bd), dtype=bool)),
            ]
            arr_descrs = []  # list of (shm_name, shape_tuple, dtype_str)
            for tag, arr in arrs:
                shm = shared_memory.SharedMemory(
                    create=True, size=arr.nbytes,
                    name=f"tract7dt_psf_{bn}_{tag}_{_pid}",
                )
                # Publish the handle BEFORE the fill: if a later allocation
                # or copy fails mid-stage, the cleanup `finally` below must
                # already see every segment created so far. (list.append is
                # atomic under the GIL, so the copy threads can share the
                # list safely.)
                shm_handles.append(shm)
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
                arr_descrs.append((shm.name, tuple(arr.shape), str(arr.dtype)))
            rows_b, limit_b, _ = training[bn]
            task = (
                bn,
                arr_descrs[0][0], arr_descrs[0][1], arr_descrs[0][2],  # img
                arr_descrs[1][0], arr_descrs[1][1], arr_descrs[1][2],  # sig
                arr_descrs[2][0], arr_descrs[2][1], arr_descrs[2][2],  # bad
                arr_descrs[3][0], arr_descrs[3][1], arr_descrs[3][2],  # sat
                ps_cfg,
                str(out_dir), save_diag,
                rows_b, limit_b,
            )
            return task

        results_list = []
        try:
            # The upfront shm fill is dominated by the ~480 MB/band memcpy (~9.6 GB
            # total) which releases the GIL, so copy bands concurrently across
            # threads instead of serially. map() preserves order, so tasks stay in
            # bandnames order; the bytes landed in shm are identical (bit-identical).
            # INSIDE the try: any mid-allocation failure (ENOSPC, FileExistsError)
            # must reach the cleanup `finally`, or the already-created segments
            # leak into /dev/shm until the next run's sweep.
            copy_workers = min(len(bandnames), max(1, os.cpu_count() or 1), 8)
            with ThreadPoolExecutor(max_workers=copy_workers) as _cp:
                tasks = list(_cp.map(_alloc_band_shm, bandnames))

            n_blas = blas_threads_for(max_workers)
            logger.info("psf_from_stars: %d worker(s), BLAS threads per worker %d "
                        "(physical cores in affinity // workers)", max_workers, n_blas)
            with pinned_blas_env(n_blas), ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_psf_worker_init,
            ) as pool:
                futures = {pool.submit(_process_band_shm, *t): t[0] for t in tasks}
                for fut in as_completed(futures):
                    bn_done, summary_or_result, err = fut.result()
                    results_list.append((bn_done, summary_or_result, err))
                    if summary_or_result is not None and isinstance(summary_or_result, dict):
                        # Worker already wrote files + returned a summary;
                        # log immediately so the user sees per-band progress
                        # in the same order the previous pipeline did.
                        logger.info(
                            "psf_from_stars: band %s done (n_used=%d, "
                            "fwhm_mode=%.2f, %.1fs)",
                            bn_done, summary_or_result["n_used"],
                            summary_or_result["fwhm_mode"],
                            summary_or_result["timing_sec"],
                        )
            bn_to_res = {bn: (bn, r, e) for bn, r, e in results_list}
            results_list = [bn_to_res[bn] for bn in bandnames]
        finally:
            # Always release shared memory, even on exceptions.
            for shm in shm_handles:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass

    for bn, result_or_summary, err in results_list:
        if result_or_summary is None:
            logger.error("psf_from_stars: band %s FAILED: %s", bn, err)
            summary_rows.append({"band": bn, "status": "failed", "error": err})
            continue
        if isinstance(result_or_summary, dict):
            # Worker-side IO path (max_workers > 1): just append summary.
            s = result_or_summary
            _log_integral_qc(bn, s)
            if save_diag:
                _submit_psf_residual_montage(out_dir, bn)
            summary_rows.append({
                "band": bn, "status": "ok",
                "n_used": s["n_used"],
                "n_after_gaia": s["n_after_gaia"],
                "n_after_quality": s["n_after_quality"],
                "fwhm_mode_pix": s["fwhm_mode"],
                "outer_iters": s["outer_iters"],
                "timing_sec": s["timing_sec"],
            })
            continue
        # In-process serial path (max_workers == 1): we still hold the
        # full PsfFromStarsResult, so do the IO here.
        result = result_or_summary
        _log_integral_qc(bn, _save_one_psf(out_dir, bn, result, save_diag))
        if save_diag:
            try:
                _save_psf_residual_montage(
                    out_dir / bn / "psf_residual.png", bn, result, n_show=25,
                )
            except Exception as plot_exc:
                logger.warning("psf_from_stars: failed to draw residual "
                               "montage for %s: %s", bn, plot_exc)
        summary_rows.append({
            "band": bn, "status": "ok",
            "n_used": result.n_stars_used_in_fit,
            "n_after_gaia": result.n_stars_after_gaia,
            "n_after_quality": result.n_stars_after_quality,
            "fwhm_mode_pix": result.fwhm_mode,
            "outer_iters": result.n_outer_iters,
            "timing_sec": result.timing_sec,
        })
        logger.info(
            "psf_from_stars: band %s done (n_used=%d, fwhm_mode=%.2f, %.1fs)",
            bn, result.n_stars_used_in_fit, result.fwhm_mode, result.timing_sec,
        )

    for row in summary_rows:
        if row.get("status") == "ok":
            with open(out_dir / row["band"] / "cache_key.json", "w") as f:
                json.dump(cache_keys[row["band"]], f, indent=1, sort_keys=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    n_ok = int((summary_df["status"] == "ok").sum())
    logger.info(
        "psf_from_stars: stage done in %.1fs (%d/%d bands successful)",
        time.perf_counter() - t_start, n_ok, n_bands,
    )

    state["psf_polynomial_dir"] = str(out_dir)
    return out_dir


def load_psf_polynomial_for_band(
    psf_dir: Path, band: str,
):
    """Reverse of :func:`_save_one_psf`. Returns a
    :class:`PsfPolynomial` (numpy-side; caller does ``.to_jax()``)."""
    from .psf_polynomial import PsfPolynomial
    npz_path = Path(psf_dir) / band / "psf_polynomial.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"PSF polynomial not found for band {band}: {npz_path}")
    data = np.load(npz_path)
    return PsfPolynomial(
        coefs=data["coefs"].astype(np.float32),
        x_origin=float(data["x_origin"]),
        x_scale=float(data["x_scale"]),
        y_origin=float(data["y_origin"]),
        y_scale=float(data["y_scale"]),
        H_psf=int(data["H_psf"]),
        W_psf=int(data["W_psf"]),
    )


__all__ = [
    "build_psf_polynomial_from_stars_stage",
    "load_psf_polynomial_for_band",
]
