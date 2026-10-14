"""Pipeline orchestration.

This is the slim orchestrator that wires together the individual stages of
``tract7dt run``. The heavy lifting lives in dedicated modules:

- :mod:`tract7dt.load_inputs`       - frame load + white coadd + crop + overlay
- :mod:`tract7dt.bright_mask`       - bright-source masking
- :mod:`tract7dt.fit.grid_gpu`      - PSF from stars + grid/island GPU fit
- :mod:`tract7dt.zp`                - GaiaXP augmentation + ZP calibration

This module exposes the end-to-end ``run_pipeline``.
"""

from __future__ import annotations

import logging
import multiprocessing as _mp
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd

from .load_inputs import load_inputs

logger = logging.getLogger("tract7dt.pipeline")


def _log(msg: str) -> None:
    text = str(msg).strip()
    if text.startswith("[WARN]"):
        logger.warning(text[len("[WARN]") :].strip())
    elif text.startswith("[CHECK]") and "FAILED" in text:
        logger.warning(text)
    else:
        logger.info(text)


def _format_elapsed(dt: float) -> str:
    return f"{dt:.2f}s"


def _summarize_fit_results(final_catalog: Path, df: "pd.DataFrame | None" = None) -> None:
    if df is None:
        if not final_catalog.exists():
            _log(f"[WARN] Final catalog not found for summary: {final_catalog}")
            return
        try:
            df = pd.read_csv(final_catalog)
        except Exception as e:
            _log(f"[WARN] Could not read final catalog for summary: {e}")
            return
    if "opt_converged" not in df.columns or "opt_hit_max_iters" not in df.columns:
        _log("[WARN] Summary skipped: opt_converged/opt_hit_max_iters missing.")
        return
    conv = pd.to_numeric(df["opt_converged"], errors="coerce").fillna(0).astype(int).sum()
    hit = pd.to_numeric(df["opt_hit_max_iters"], errors="coerce").fillna(0).astype(int).sum()
    _log(f"Fit summary: converged={conv}, hit_max_iters={hit}, total={len(df)}")


def _compose_final_masks(state: dict) -> None:
    """Compose the final per-band ``bad`` mask from non-finite | saturation | bright."""
    image_dict = state["image_dict"]
    bright_mask = state.get("bright_mask", None)
    for band, d in image_dict.items():
        # R30: keep the no-data part (non-finite | outside the footprint),
        # bit-packed (~9 MB/band), for the partial-coverage band flag.
        d["nodata_bits"] = np.packbits(np.asarray(d["bad"], dtype=bool), axis=1)
        composed = np.asarray(d["bad"], dtype=bool) | np.asarray(d["satur_mask"], dtype=bool)
        if bright_mask is not None:
            composed = composed | bright_mask
        d["bad"] = composed
    n_bands = len(image_dict)
    has_bright = bright_mask is not None and np.any(bright_mask)
    _log(f"Composed final bad masks for {n_bands} bands (bright_mask={'yes' if has_bright else 'no'})")


def run_pipeline(cfg: dict) -> None:
    # A config that did not come from load_config/resolve_config (a hand-built
    # dict) gets the same defaults, retired-key rules and checks as the CLI;
    # input paths resolve against the current directory. An already resolved
    # config passes through untouched (card 018, design_v2 5.1).
    from .config import is_resolved_config, resolve_config
    if not is_resolved_config(cfg):
        cfg = resolve_config(cfg, Path.cwd())
    # R13: which code, configuration and environment produce this run's
    # outputs.
    from .provenance import write_run_provenance
    _log(f"Run provenance: {write_run_provenance(cfg, Path(cfg['outputs']['work_dir']))}")

    log_cfg = cfg.get("logging", {})
    ignore_warnings = bool(log_cfg.get("ignore_warnings", True))
    if ignore_warnings:
        warnings.filterwarnings("ignore")
        _log("Warnings are suppressed (logging.ignore_warnings=true).")
    else:
        warnings.filterwarnings("default")
        _log("Warnings are enabled.")

    # Reclaim stale /dev/shm segments from previous CRASHED runs FIRST,
    # before load_inputs' multi-GB allocations. Leaked segments are tmpfs-
    # backed RAM; sweeping only at the grid/PSF stage start would let a
    # big enough leak OOM-kill this run during input loading -- recreating
    # the crash->leak->starve cascade the sweep exists to break. The
    # module is import-light (stdlib only) and a clean scan costs ~40 ms.
    from .fit.grid_gpu.shm_hygiene import sweep_stale_segments
    sweep_stale_segments(logger)

    # Optional background plot pool. Pre-EPSF stages (load_inputs,
    # bright_mask, zp.augment) each emit a few full-frame diagnostic
    # PNGs whose results are write-only - no downstream stage reads
    # them - so we can dispatch the matplotlib savefigs to a thread
    # pool and let the next stage start immediately. The pool is
    # drained in the finally block so all PNGs are on disk by the time
    # the pipeline call returns.
    perf_cfg = cfg.get("performance", {})
    use_async_plots = bool(perf_cfg.get("background_plotting", True))
    if use_async_plots:
        from .async_plot import init_plot_pool, shutdown_plot_pool
        init_plot_pool(max_workers=int(perf_cfg.get("background_plot_workers", 4)))
    else:
        shutdown_plot_pool = None  # type: ignore[assignment]

    t_total = time.perf_counter()
    bright_pool = None  # side-process SEP pool; see below (referenced in finally)

    from . import run_status

    def _run_step(name: str, func):
        run_status.set_stage(name)
        _log(f"Starting {name}...")
        t0 = time.perf_counter()
        result = func()
        _log(f"Finished {name} in {_format_elapsed(time.perf_counter() - t0)}")
        return result

    try:
        state = _run_step("load inputs + validation", lambda: load_inputs(cfg))

        # Bright-mask SEP detection: ~66-75 s, single-core, and independent of
        # the catalog (it reads only the white stack). In gpu mode run it in a
        # SPAWNED SIDE PROCESS so it overlaps the Gaia-augment + PSF-from-stars
        # stages (~55 s hidden, measured); finalize_bright_mask (catalog match
        # + mask build, ~2 s) joins it just before mask composition. An
        # in-process background THREAD was tried before and measured slower --
        # sep.extract and the augment stage contend for the one GIL -- but a
        # separate process sidesteps the GIL entirely. If the side process
        # cannot start, the sequential path below runs instead.
        bright_enabled = bool(cfg.get("bright_mask", {}).get("enabled", True))
        bright_fut = None
        bright_objs_cached = None
        if bright_enabled:
            # T4-remainder (2026-08-15, development ledger SS18): the SEP
            # detection input (white/sigma_white) is pass-invariant and
            # phantom-independent, so a prior pass's objs are adopted under
            # a HARD digest key instead of re-running the side process --
            # whose only hiding window on the phantom pass B (the PSF build)
            # the PSF npz cache removed, leaving a ~100 s exposed join wait
            # on crowded fields. A key miss or any error spawns as before.
            try:
                from .bright_mask import load_sep_objs_cache
                bright_objs_cached = load_sep_objs_cache(cfg, state)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SEP objs cache check failed (%s); spawning detection",
                    exc)
                bright_objs_cached = None
        if bright_enabled and bright_objs_cached is None:
            try:
                from .bright_mask import detect_bright_sources_standalone
                from .fit.grid_gpu.shm_hygiene import start_parent_watchdog
                bright_pool = _ProcessPoolExecutor(
                    max_workers=1, mp_context=_mp.get_context("spawn"),
                    initializer=start_parent_watchdog,
                )
                bright_fut = bright_pool.submit(
                    detect_bright_sources_standalone,
                    cfg, state["white"], state["sigma_white"],
                )
                _log("bright-mask SEP detection started in a side process "
                     "(overlaps Gaia augment + PSF stages)")
            except Exception as exc:  # noqa: BLE001 - fall back to sequential
                logger.warning("could not start background bright-mask "
                               "detection (%s); running sequentially", exc)
                bright_pool = None
                bright_fut = None

        if bool(cfg.get("zp", {}).get("enabled", False)):
            from .zp import augment_catalog_with_gaia
            _run_step("augment catalog with Gaia sources", lambda: augment_catalog_with_gaia(cfg=cfg, state=state))

        if bright_enabled and bright_fut is None and bright_objs_cached is None:
            from .bright_mask import apply_bright_mask
            _run_step("apply bright-source mask", lambda: apply_bright_mask(cfg, state))

        # IMPORTANT ORDERING: psf.from_stars MUST run BEFORE
        # _compose_final_masks. The bright-source mask is composed
        # INTO state['image_dict'][band]['bad'] by _compose_final_masks,
        # which would mask out all our bright PSF stars from SEP
        # detection. Building the PSF first sees only the original
        # saturation mask (unaffected by bright_mask).
        from .fit.grid_gpu.psf_pipeline_stage import (
            build_psf_polynomial_from_stars_stage,
        )
        _run_step(
            "build psf polynomial from stars (PSFEx-style)",
            lambda: build_psf_polynomial_from_stars_stage(cfg, state),
        )
        if bright_fut is not None or bright_objs_cached is not None:
            # Join the side-process SEP detection (usually long done by
            # now) -- or adopt the cached objs -- and run the
            # catalog-dependent finalize. Must precede
            # _compose_final_masks, which folds the mask into 'bad'.
            from .bright_mask import detect_bright_sources, finalize_bright_mask

            def _finish_bright_mask():
                if bright_objs_cached is not None:
                    return finalize_bright_mask(
                        cfg, state, bright_objs_cached)
                _tj0 = time.perf_counter()
                try:
                    objs = bright_fut.result()
                    # A0 (LEDGER SS18): the side process logs into its
                    # own spawn context, so this join line is the only
                    # in-log record of what the SEP actually cost us.
                    _log(f"bright-mask SEP side process joined in "
                         f"{time.perf_counter() - _tj0:.1f}s")
                except Exception as exc:  # noqa: BLE001 - worker died
                    logger.warning(
                        "background bright-mask detection failed (%s); "
                        "rerunning inline", exc)
                    objs = detect_bright_sources(cfg, state)
                finally:
                    bright_pool.shutdown(wait=False)
                try:
                    from .bright_mask import save_sep_objs_cache
                    save_sep_objs_cache(cfg, state, objs)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SEP objs cache save failed: %s", exc)
                return finalize_bright_mask(cfg, state, objs)

            _run_step("finalize bright-source mask", _finish_bright_mask)
        _run_step("compose final masks", lambda: _compose_final_masks(state))
        # R29: the ZP calibrator quality flags are in hand before the
        # fit, or the run fails here (not after the GPU work).
        from .zp import require_gaia_quality_flags
        _run_step("obtain Gaia quality flags",
                  lambda: require_gaia_quality_flags(cfg, state))
        from .fit.grid_gpu.pipeline import run_grid_gpu_fit
        final_catalog = _run_step(
            "run gpu fit",
            lambda: run_grid_gpu_fit(cfg, state),
        )
        # In gpu mode the grid stage may hand the catalog to us in memory and
        # defer the output_catalog.csv write to compute_zp (saves a redundant
        # ~443 MB write + re-read). state["gpu_output_df"] is None otherwise.
        gpu_df = state.get("gpu_output_df")
        _summarize_fit_results(Path(final_catalog), df=gpu_df)

        # With zp.enabled the grid stage hands the catalog over in memory and
        # compute_zp writes it; without, the grid stage wrote it itself.
        if bool((cfg.get("zp", {}) or {}).get("enabled", False)):
            from .zp import compute_zp
            _gaia_syn = state.get("_gaia_synphot_df")
            _run_step("compute zero-point calibration",
                      lambda: compute_zp(cfg=cfg, merged_catalog_path=Path(final_catalog),
                                         catalog_df=gpu_df, gaia_synphot_df=_gaia_syn))
        state.pop("_gaia_synphot_df", None)  # free ~560 MB synphot frame
        state.pop("gpu_output_df", None)  # free ~443 MB

        _log(f"Pipeline finished in {_format_elapsed(time.perf_counter() - t_total)}")
    finally:
        if bright_pool is not None:
            # Normally already shut down by _finish_bright_mask; this covers
            # exceptions raised between the spawn and the finalize step.
            bright_pool.shutdown(wait=False, cancel_futures=True)
        try:
            from .fit.grid_gpu.psf_pipeline_stage import shutdown_psf_diagnostics
            shutdown_psf_diagnostics()
        except Exception as exc:
            logger.warning("PSF diagnostic background shutdown failed: %s", exc)
        if shutdown_plot_pool is not None:
            shutdown_plot_pool()
