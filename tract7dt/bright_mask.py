"""
White-stack bright unmatched-source masking for tract7dt.

Runs SEP on the in-memory white stack, matches detections against active
catalog sources, and builds ellipse-based masks for unmatched bright
contaminants.  The resulting mask is stored in ``state["bright_mask"]``
and later composed with saturation and non-finite masks into per-band
``bad`` arrays by the pipeline's ``_compose_final_masks`` step, so that
the downstream fit sees ``invvar = 0`` for those pixels.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")

from .sep_limits import apply_sep_limits
from .utils import case_insensitive_col_map
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

logger = logging.getLogger("tract7dt.bright_mask")


def detect_bright_sources(cfg: dict[str, Any], state: dict[str, Any]):
    """Run SEP detection on the in-memory white stack and return the raw SEP
    ``objs`` array (or ``None`` if bright masking is disabled).

    This reads ONLY ``state["white"]`` / ``state["sigma_white"]`` (each is
    ``.copy()``-ed before mutation) and writes nothing into ``state``, so it is
    safe to run in a background thread overlapping the Gaia-augment and
    PSF-from-stars stages (neither of which touches the white stack). ``sep``'s C
    detection loop releases the GIL, so the overlap is real. The catalog-matching,
    mask-building, source-flagging, and diagnostics live in
    :func:`finalize_bright_mask`, which must run after the augmented catalog is
    ready and after this detection has been joined.
    """
    bm_cfg = cfg.get("bright_mask", {})
    if not bool(bm_cfg.get("enabled", True)):
        logger.info("bright_mask disabled, skipping.")
        return None

    white = np.asarray(state["white"], dtype=np.float32)
    sigma_white = np.asarray(state["sigma_white"], dtype=np.float32)

    # --- SEP detection on white stack ---
    white_sep = white.copy()
    bad_pix = ~np.isfinite(white_sep)
    white_sep[bad_pix] = 0.0

    sig_sep = sigma_white.copy()
    sig_bad = ~np.isfinite(sig_sep) | (sig_sep <= 0)
    sig_sep[sig_bad] = 1e30

    p = sep_extract_params(cfg)
    thresh = p["thresh_sigma"]
    # Extreme-crowding fields (|b|~2 bulge, 237k-src tile) overflow SEP's
    # default limits at the shipped threshold (2026-07-31 lowb trial finding
    # #5). The package-wide limits (sep_limits.py) only matter when the call
    # would otherwise raise.
    apply_sep_limits(sep)
    objs = sep.extract(
        white_sep,
        thresh=thresh,
        err=sig_sep,
        mask=bad_pix.astype(np.uint8),
        minarea=p["minarea"],
        deblend_nthresh=p["deblend_nthresh"],
        deblend_cont=p["deblend_cont"],
        clean=p["clean"],
        clean_param=p["clean_param"],
    )
    logger.info("SEP detected %d objects on white stack (thresh=%.1f sigma)", len(objs), thresh)
    return objs


def sep_extract_params(cfg: dict[str, Any]) -> dict:
    """The EXACT parameter set the SEP detection call consumes -- the single
    source for both the call above and the pass-to-pass objs cache key
    (T4-remainder, 2026-08-15). A knob added to the call must be read here,
    so it enters the cache key in the same edit."""
    bm_cfg = cfg.get("bright_mask", {}) or {}
    return {
        "thresh_sigma": float(bm_cfg.get("thresh_sigma", 7.0)),
        "minarea": int(bm_cfg.get("minarea", 20)),
        "deblend_nthresh": int(bm_cfg.get("deblend_nthresh", 32)),
        "deblend_cont": float(bm_cfg.get("deblend_cont", 0.005)),
        "clean": bool(bm_cfg.get("clean", True)),
        "clean_param": float(bm_cfg.get("clean_param", 1.0)),
    }


def _sep_cache_key(cfg: dict[str, Any], state: dict[str, Any]) -> dict:
    """HARD key for the SEP objs cache: engine version + the exact SEP
    parameters + blake2b digests of the actual detection inputs (white and
    sigma_white). Digests, not mtimes/sizes: the same images still produce a
    different white under different scaling/coverage/crop settings, and the
    digest sees that where any proxy would not (~0.5 s per array)."""
    import hashlib

    from . import __version__

    white = np.asarray(state["white"], dtype=np.float32)
    sigw = np.asarray(state["sigma_white"], dtype=np.float32)
    hw = hashlib.blake2b(digest_size=16)
    hw.update(white.tobytes())
    hs = hashlib.blake2b(digest_size=16)
    hs.update(sigw.tobytes())
    return {"engine": str(__version__), "sep": sep_extract_params(cfg),
            "shape": [int(x) for x in white.shape],
            "white": hw.hexdigest(), "sigma": hs.hexdigest()}


def sep_objs_cache_path(cfg: dict[str, Any]):
    wd = (cfg.get("outputs", {}) or {}).get("work_dir")
    if not wd:
        return None
    return Path(wd) / "bright_mask" / "sep_objs.npz"


def load_sep_objs_cache(cfg: dict[str, Any], state: dict[str, Any]):
    """T4-remainder (2026-08-15, user decision; development ledger SS18):
    adopt a prior pass's SEP detections instead of re-running the side
    process. The detection reads ONLY white/sigma_white (pass-invariant --
    the phantom list is consumed by the FIT stage, never by detection), so
    under a matching hard key the cached objs ARE what a fresh run would
    produce. Any mismatch or error returns None and the caller spawns the
    detection exactly as before."""
    import json

    p = sep_objs_cache_path(cfg)
    if p is None or not p.exists():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            key = json.loads(str(z["key"][0]))
            if key != _sep_cache_key(cfg, state):
                logger.info(
                    "SEP objs cache key mismatch (inputs or params changed) "
                    "-- detecting fresh")
                return None
            objs = z["objs"]
        logger.info(
            "bright-mask SEP objs adopted from cache (%d objects; hard key: "
            "engine + SEP params + white/sigma digests) -- side process not "
            "spawned", len(objs))
        return objs
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEP objs cache unreadable (%s); detecting fresh", exc)
        return None


def save_sep_objs_cache(cfg: dict[str, Any], state: dict[str, Any],
                        objs) -> None:
    import json
    import os

    p = sep_objs_cache_path(cfg)
    if p is None or objs is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp.npz"
        np.savez(tmp, objs=objs,
                 key=np.array([json.dumps(_sep_cache_key(cfg, state))]))
        os.replace(tmp, str(p))
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEP objs cache write failed: %s", exc)


def finalize_bright_mask(cfg: dict[str, Any], state: dict[str, Any], objs) -> np.ndarray:
    """Match the SEP detections (from :func:`detect_bright_sources`) against the
    current (possibly Gaia-augmented) catalog, build + store the bright mask in
    ``state["bright_mask"]``, flag affected sources, and save diagnostics.

    Reads ``state["input_catalog"]`` fresh, so it must run AFTER any Gaia
    augmentation (which reassigns that key). Returns the 2-D boolean mask.
    """
    bm_cfg = cfg.get("bright_mask", {})
    input_catalog = state["input_catalog"]
    image_dict = state["image_dict"]
    white = np.asarray(state["white"], dtype=np.float32)
    h, w = white.shape

    if objs is None:
        # bright masking disabled (detect returned None): empty mask.
        mask = np.zeros((h, w), dtype=bool)
        input_catalog["affected_by_bright_mask"] = False
        state["bright_mask"] = mask
        return mask

    if len(objs) == 0:
        mask = np.zeros((h, w), dtype=bool)
        input_catalog["affected_by_bright_mask"] = False
        state["bright_mask"] = mask
        _save_diagnostics(cfg, white, mask, objs, np.array([], dtype=bool),
                          input_catalog, image_dict)
        return mask

    det_x = objs["x"]
    det_y = objs["y"]

    # --- Match detections to active catalog sources ---
    match_r = float(bm_cfg.get("match_radius_pix", 3.0))
    matched = _match_to_catalog(det_x, det_y, input_catalog, image_dict, match_r)
    n_matched = int(np.sum(matched))
    n_unmatched = len(objs) - n_matched
    logger.info("Matched %d / %d SEP detections to catalog; %d unmatched to mask",
                n_matched, len(objs), n_unmatched)

    if n_unmatched == 0:
        mask = np.zeros((h, w), dtype=bool)
        input_catalog["affected_by_bright_mask"] = False
        state["bright_mask"] = mask
        _save_diagnostics(cfg, white, mask, objs, matched,
                          input_catalog, image_dict)
        return mask

    # --- Convert-to-model mode: route unmatched detections into the island
    #     fit as compact n-frozen DET (soaked, unemitted) instead of masking
    #     their wings, so previously-masked neighbours stay fittable. The
    #     genuinely information-free saturated cores remain masked by the
    #     independent per-band saturation mask. Detections are stashed in the
    #     SAME cropped-white pixel frame the island fit uses (objs["x"]/["y"]
    #     share xpix/ypix's sliced WCS), so no coordinate transform is needed. ---
    if (bool(bm_cfg.get("model_unmatched", False))
            and bool(bm_cfg.get("subtract_unmatched", False))):
        raise ValueError(
            "bright_mask.model_unmatched and bright_mask.subtract_unmatched "
            "are mutually exclusive (DET free-fit routing vs subtract-only "
            "routing of the same unmatched detections)")

    if bool(bm_cfg.get("model_unmatched", False)):
        um = ~matched
        state["bright_mask_det"] = np.column_stack([
            objs["x"][um].astype(np.float64),
            objs["y"][um].astype(np.float64),
            objs["flux"][um].astype(np.float64),
        ])
        mask = np.zeros((h, w), dtype=bool)
        state["bright_mask"] = mask
        input_catalog["affected_by_bright_mask"] = False
        logger.info(
            "bright_mask: model_unmatched -> %d unmatched detection(s) routed "
            "as DET (no wing mask; saturation cores still masked)",
            int(um.sum()))
        _save_diagnostics(cfg, white, mask, objs, matched,
                          input_catalog, image_dict)
        return mask

    # --- Build mask for unmatched detections ---
    # mask_radius_mode: "ellipse" (legacy geometric ellipse_scale/min_radius/
    # dilate) or "physical" (PSF-wing surface brightness vs local noise; no
    # tuned geometric constants -- see _build_physical_mask). Physical falls
    # back to ellipse when no from_stars PSF profile is available.
    mode = str(bm_cfg.get("mask_radius_mode", "ellipse"))
    profile = _load_white_psf_profile(cfg, state) if mode == "physical" else None
    if mode == "physical" and profile is None:
        logger.warning("bright_mask mask_radius_mode='physical' but no from_stars "
                       "PSF profile available; using legacy ellipse mask")

    # --- Exp C "subtract, don't mask" routing: unmatched detections whose
    #     above-noise wing extent fits INSIDE the from_stars stamp's
    #     modelable radius are handed to the island subtract-only pass
    #     instead of being wing-masked (masking an uncataloged star
    #     annihilates nearby galaxy fits -- oracle-measured). Wings that
    #     the stamp cannot model (radius clamped at the profile edge) keep
    #     the mask; saturated cores stay masked per-band regardless. ---
    keep_unmatched = ~matched
    _route_requested = bool(bm_cfg.get("subtract_unmatched", False))
    if _route_requested:
        _pop = str(bm_cfg.get("unmatched_population", "stars")).lower()
        if _pop != "stars":
            # USER DECLARATION: this field's uncataloged detections are
            # (partly) extended. No dedicated extended treatment exists
            # yet -> FALL BACK to point-source modeling (the only
            # implemented form) and say so loudly. The on/off gate is
            # subtract_unmatched alone -- this knob never silently
            # disables the feature. NOTE (measured, UDS deep field):
            # point-source modeling of extended uncataloged sources has a
            # heavy churn tail; for deep extragalactic fields the
            # recommendation is to keep subtract_unmatched OFF.
            logger.warning(
                "bright_mask.subtract_unmatched: unmatched_population=%r "
                "declared, but a dedicated extended treatment is not yet "
                "implemented -- FALLING BACK to point-source modeling. "
                "Measured caveat: this has a heavy photometry-churn tail "
                "on galaxy-dominated fields (UDS: 40%% of sources move "
                ">10 mmag); for deep extragalactic fields consider "
                "keeping subtract_unmatched off.", _pop)
    if _route_requested:
        comp = cfg.get("compute", {}) or {}
        sol = (comp.get("gpu", {}) or {}).get("solver", {}) or {}
        gpu_ok = (str(sol.get("position_solver", "")) == "island"
                  and bool((sol.get("island", {}) or {}).get(
                      "scene_subtract", True)))
        if profile is None:
            logger.warning(
                "bright_mask.subtract_unmatched needs the physical-mask PSF "
                "profile (from_stars); falling back to masking")
        elif not gpu_ok:
            logger.warning(
                "bright_mask.subtract_unmatched requires compute.mode=gpu + "
                "position_solver=island + island.scene_subtract; falling "
                "back to masking")
        else:
            nsig_r = float(bm_cfg.get("physical_nsigma", 1.0))
            sw = np.asarray(state["sigma_white"], dtype=np.float32)
            um_idx = np.where(~matched)[0]
            r_all = _physical_radii(objs, um_idx, sw,
                                    profile[0], profile[1], nsig_r)
            r_model = float(profile[0][-1])
            routable = (r_all > 0.0) & (r_all < r_model * (1.0 - 1e-9))
            route_idx = um_idx[routable]
            if len(route_idx):
                state["bright_mask_sub"] = np.column_stack([
                    objs["x"][route_idx].astype(np.float64),
                    objs["y"][route_idx].astype(np.float64),
                    objs["flux"][route_idx].astype(np.float64)])
                state["bright_mask_sub_nsigma"] = float(
                    bm_cfg.get("thresh_sigma", 5.0))
                keep_unmatched = keep_unmatched.copy()
                keep_unmatched[route_idx] = False
            logger.info(
                "bright_mask: subtract_unmatched routed %d/%d unmatched "
                "detection(s) to the island subtract-only pass (wing radius "
                "< stamp radius %.1f px); %d keep the wing mask",
                len(route_idx), len(um_idx), r_model,
                int(keep_unmatched.sum()))

    if profile is not None:
        nsigma = float(bm_cfg.get("physical_nsigma", 1.0))
        sigma_white = np.asarray(state["sigma_white"], dtype=np.float32)
        mask = _build_physical_mask(
            h, w, objs, keep_unmatched, sigma_white, profile[0], profile[1],
            nsigma,
        )
        logger.info("bright_mask: physical-radius mode (nsigma=%.2g)", nsigma)
    else:
        ellipse_scale = float(bm_cfg.get("ellipse_scale", 1.5))
        min_radius = float(bm_cfg.get("min_radius_pix", 6.0))
        dilate_pix = int(bm_cfg.get("dilate_pix", 2))
        mask = _build_ellipse_mask(
            h, w, objs, keep_unmatched, ellipse_scale, min_radius,
        )
        if dilate_pix > 0:
            struct = np.ones((2 * dilate_pix + 1, 2 * dilate_pix + 1), dtype=bool)
            mask = binary_dilation(mask, structure=struct).astype(bool)

    n_masked_pix = int(np.sum(mask))
    logger.info("Bright-source mask: %d pixels (%.2f%% of frame)",
                n_masked_pix, 100.0 * n_masked_pix / (h * w))

    # --- Store mask in state (composed into bad later by _compose_final_masks) ---
    state["bright_mask"] = mask
    logger.info("Stored bright mask in state (will be composed into bad later)")

    # --- Flag catalog sources whose positions fall inside the mask ---
    _flag_affected_sources(input_catalog, image_dict, mask)

    # --- Save diagnostics ---
    _save_diagnostics(cfg, white, mask, objs, matched,
                      input_catalog, image_dict)

    return mask


def detect_bright_sources_standalone(
    cfg: dict[str, Any], white: np.ndarray, sigma_white: np.ndarray,
):
    """Spawn-friendly wrapper around :func:`detect_bright_sources`: takes the
    bare arrays instead of the state dict, so the gpu pipeline can run the
    ~66 s single-core SEP detection in a SIDE PROCESS overlapping the
    Gaia-augment + PSF stages. (An in-process background THREAD was tried
    before and measured slower -- sep.extract and the augment stage contend
    for the one GIL; a separate process sidesteps that entirely.) Identical
    inputs -> identical detections, so the mask is the same either way."""
    return detect_bright_sources(cfg, {"white": white, "sigma_white": sigma_white})


def apply_bright_mask(cfg: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
    """Detect bright unmatched sources on the white stack, build ellipse masks,
    store in ``state["bright_mask"]``, and save diagnostics.

    Sequential detect+finalize wrapper (used by cpu/legacy_gpu mode and the
    partial-rerun CLI commands). The gpu pipeline runs
    :func:`detect_bright_sources_standalone` in a spawned side process and
    :func:`finalize_bright_mask` after the join, to overlap the ~66 s SEP
    detection with the Gaia-augment + PSF stages. Both paths produce an
    identical mask. Returns the 2-D boolean mask (True = masked).
    """
    objs = detect_bright_sources(cfg, state)
    return finalize_bright_mask(cfg, state, objs)


def _match_to_catalog(
    det_x: np.ndarray,
    det_y: np.ndarray,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, Any],
    match_radius_pix: float,
) -> np.ndarray:
    """Return a boolean array of length len(det_x), True where a detection
    matches an active catalog source within match_radius_pix."""

    is_active = ~input_catalog.get("excluded_any", pd.Series(False, index=input_catalog.index)).fillna(False).astype(bool)
    active_cat = input_catalog.loc[is_active]

    if len(active_cat) == 0:
        return np.zeros(len(det_x), dtype=bool)

    wcs_white = next(iter(image_dict.values()))["wcs"]
    col_map = case_insensitive_col_map(active_cat.columns)
    ra_col = col_map.get("ra", "RA")
    dec_col = col_map.get("dec", "DEC")

    ra = pd.to_numeric(active_cat[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(active_cat[dec_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)

    if not np.any(ok):
        return np.zeros(len(det_x), dtype=bool)

    cat_x = np.full(len(ra), np.nan, dtype=float)
    cat_y = np.full(len(dec), np.nan, dtype=float)
    xw, yw = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
    cat_x[ok] = xw
    cat_y[ok] = yw

    cat_x = cat_x[ok]
    cat_y = cat_y[ok]

    if len(det_x) == 0:
        return np.zeros(0, dtype=bool)

    if os.environ.get("TRACT7DT_BENCH_DISABLE_KDTREE") == "1":
        matched = np.zeros(len(det_x), dtype=bool)
        r2 = match_radius_pix ** 2
        for i in range(len(det_x)):
            dx = det_x[i] - cat_x
            dy = det_y[i] - cat_y
            if np.any(dx * dx + dy * dy <= r2):
                matched[i] = True
        return matched

    cat_xy = np.column_stack([cat_x, cat_y])
    det_xy = np.column_stack([
        np.asarray(det_x, dtype=float),
        np.asarray(det_y, dtype=float),
    ])
    det_ok = np.isfinite(det_xy).all(axis=1)
    matched = np.zeros(len(det_x), dtype=bool)
    if np.any(det_ok):
        tree = cKDTree(cat_xy)
        hits = tree.query_ball_point(det_xy[det_ok], r=float(match_radius_pix))
        matched[det_ok] = np.fromiter((len(h) > 0 for h in hits), dtype=bool)

    return matched


def _flag_affected_sources(
    input_catalog: pd.DataFrame,
    image_dict: dict[str, Any],
    mask: np.ndarray,
) -> None:
    """Flag catalog sources whose pixel positions fall inside the bright mask."""
    input_catalog["affected_by_bright_mask"] = False

    wcs_white = next(iter(image_dict.values()))["wcs"]
    col_map = case_insensitive_col_map(input_catalog.columns)
    ra_col = col_map.get("ra", "RA")
    dec_col = col_map.get("dec", "DEC")

    ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)

    if not np.any(ok):
        return

    cat_x = np.full(len(ra), np.nan, dtype=float)
    cat_y = np.full(len(dec), np.nan, dtype=float)
    xw, yw = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
    cat_x[ok] = xw
    cat_y[ok] = yw

    h, w = mask.shape
    # Vectorized equivalent of the per-source loop: round to nearest pixel,
    # keep in-bounds finite positions, sample the mask. Bit-identical to the
    # int(np.round(...)) + bounds-check + mask[iy, ix] loop.
    hit = np.zeros(len(input_catalog), dtype=bool)
    idx_ok = np.where(ok)[0]
    if idx_ok.size:
        ix = np.round(cat_x[idx_ok]).astype(np.int64)
        iy = np.round(cat_y[idx_ok]).astype(np.int64)
        inb = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        sel = idx_ok[inb]
        hit[sel] = mask[iy[inb], ix[inb]]
    input_catalog["affected_by_bright_mask"] = hit

    n_affected = int(input_catalog["affected_by_bright_mask"].sum())
    if n_affected > 0:
        logger.info("Flagged %d catalog sources as affected_by_bright_mask", n_affected)


def _build_ellipse_mask(
    h: int,
    w: int,
    objs: np.ndarray,
    unmatched: np.ndarray,
    ellipse_scale: float,
    min_radius: float,
) -> np.ndarray:
    """Build a boolean mask from scaled SEP ellipses of unmatched detections."""
    mask = np.zeros((h, w), dtype=np.bool_)

    idx = np.where(unmatched)[0]
    if len(idx) == 0:
        return mask

    xs = objs["x"][idx].astype(np.float64)
    ys = objs["y"][idx].astype(np.float64)
    as_ = np.maximum(objs["a"][idx].astype(np.float64) * ellipse_scale, min_radius)
    bs = np.maximum(objs["b"][idx].astype(np.float64) * ellipse_scale, min_radius)
    thetas = objs["theta"][idx].astype(np.float64)

    sep.mask_ellipse(mask, xs, ys, as_, bs, thetas)
    return mask


def _load_white_psf_profile(cfg: dict[str, Any], state: dict[str, Any]):
    """Load a representative-band ``from_stars`` PSF and return its monotone
    unit-flux radial profile ``(psf_r, psf_p)`` for the physical-radius mask.

    Returns ``None`` when no from_stars PSF is available (cpu / legacy modes,
    or the PSF stage did not run), so the caller falls back to the ellipse
    mask and byte-for-byte reproduces the legacy behaviour. The PSF stage
    (``build_psf_polynomial_from_stars_stage``) runs BEFORE
    :func:`finalize_bright_mask`, so ``state['psf_polynomial_dir']`` is set by
    the time this is called in gpu mode.
    """
    psf_dir = state.get("psf_polynomial_dir")
    if not psf_dir:
        return None
    try:
        from pathlib import Path as _Path
        from .fit.grid_gpu.psf_pipeline_stage import load_psf_polynomial_for_band
        from .fit.grid_gpu.psf_polynomial import evaluate_psf_at_np
        from .fit.grid_gpu.islands import psf_radial_profile

        bands = list(state["image_dict"].keys())
        if not bands:
            return None
        band = bands[len(bands) // 2]          # middle band ~ white-stack representative
        white = np.asarray(state["white"])
        h, w = white.shape
        psf = load_psf_polynomial_for_band(_Path(str(psf_dir)), band)
        stamp = evaluate_psf_at_np(psf, w / 2.0, h / 2.0)
        psf_r, psf_p = psf_radial_profile(np.asarray(stamp, dtype=float))
        return psf_r, psf_p
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.warning("physical bright-mask: could not load PSF profile (%s); "
                       "falling back to ellipse mask", e)
        return None


def _physical_radii(
    objs: np.ndarray,
    idx: np.ndarray,
    sigma_white: np.ndarray,
    psf_r: np.ndarray,
    psf_p: np.ndarray,
    nsigma: float,
) -> np.ndarray:
    """Per-detection physical wing radius (the ``_build_physical_mask``
    radius law, factored out): the radius where the detection's flux spread
    over the from_stars PSF profile drops to ``nsigma`` x the local noise.
    Returns one radius per entry of ``idx``; 0 where below noise everywhere,
    CLAMPED to ``psf_r[-1]`` where the wings extend beyond the profile."""
    from .fit.grid_gpu.islands import _invert_monotone_profile

    h, w = np.asarray(sigma_white).shape
    xs = objs["x"][idx].astype(np.float64)
    ys = objs["y"][idx].astype(np.float64)
    flux = np.maximum(objs["flux"][idx].astype(np.float64), 0.0)

    ix = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
    iy = np.clip(np.round(ys).astype(np.int64), 0, h - 1)
    sig = np.asarray(sigma_white, dtype=np.float64)[iy, ix]
    finite = np.isfinite(sig) & (sig > 0)
    med = float(np.median(sig[finite])) if np.any(finite) else 1.0
    sig = np.where(finite, sig, med)

    with np.errstate(divide="ignore", invalid="ignore"):
        thresh = np.where(flux > 0, nsigma * sig / flux, np.inf)
    return _invert_monotone_profile(psf_r, psf_p, thresh)


def _build_physical_mask(
    h: int,
    w: int,
    objs: np.ndarray,
    unmatched: np.ndarray,
    sigma_white: np.ndarray,
    psf_r: np.ndarray,
    psf_p: np.ndarray,
    nsigma: float,
) -> np.ndarray:
    """Physics-based bright mask (no geometric tuned constants).

    For each unmatched detection, mask out to the radius where the source's
    own light -- its isophotal flux spread over the run's OWN from_stars PSF --
    still exceeds ``nsigma`` times the LOCAL per-pixel noise (``sigma_white``).
    Formally ``r_mask`` solves ``flux * psf_p(r) = nsigma * sigma_local`` via
    the monotone PSF profile: a bright contaminant has detectable wings far
    out (large mask); a faint 5-sigma blip drops below noise immediately
    (mask collapses to ~0). This replaces the arbitrary
    ``ellipse_scale``/``min_radius_pix``/``dilate_pix`` geometry with a
    forward propagation of a MEASURED flux through a KNOWN PSF versus KNOWN
    noise -- no dataset-tuned constants (``nsigma`` is a principled
    surface-brightness significance, the analogue of the SEP detection floor).
    """
    from .fit.grid_gpu.islands import _invert_monotone_profile

    mask = np.zeros((h, w), dtype=np.bool_)
    idx = np.where(unmatched)[0]
    if len(idx) == 0:
        return mask

    xs = objs["x"][idx].astype(np.float64)
    ys = objs["y"][idx].astype(np.float64)
    r_mask = _physical_radii(objs, idx, sigma_white, psf_r, psf_p, nsigma)

    keep = np.isfinite(r_mask) & (r_mask > 0.0)
    if not np.any(keep):
        return mask
    rr = r_mask[keep].astype(np.float64)
    sep.mask_ellipse(mask, xs[keep], ys[keep], rr, rr, np.zeros_like(rr))
    return mask


def _save_diagnostics(
    cfg: dict[str, Any],
    white: np.ndarray,
    mask: np.ndarray,
    objs: np.ndarray,
    matched: np.ndarray,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, Any],
) -> None:
    """Save bright-mask overlay PNG and raw FITS mask."""
    outputs = cfg.get("outputs", {})
    work_dir = outputs.get("work_dir")
    if work_dir is None:
        logger.warning("Cannot save bright-mask diagnostics: no work_dir")
        return

    bm_dir = Path(work_dir) / "bright_mask"
    bm_dir.mkdir(parents=True, exist_ok=True)

    # --- FITS mask (bright ellipses | any-band saturation) ---
    combined_mask = mask.copy()
    for d in image_dict.values():
        sm = d.get("satur_mask")
        if sm is not None:
            combined_mask = combined_mask | np.asarray(sm, dtype=bool)
    fits_path = bm_dir / "white_bright_mask.fits"
    try:
        hdu = fits.PrimaryHDU(data=combined_mask.astype(np.uint8))
        hdu.header["EXTNAME"] = "BRIGHT_MASK"
        hdu.header["COMMENT"] = "1 = masked (bright unmatched source OR saturated pixel)"
        hdu.writeto(str(fits_path), overwrite=True)
        logger.info("Saved bright-mask FITS (incl. saturation): %s", fits_path)
    except Exception as e:
        logger.warning("Failed to write bright-mask FITS: %s", e)

    # --- Overlay PNG (gated; matplotlib on full frame is ~10-20s) ---
    bm_cfg_for_overlay = cfg.get("bright_mask", {})
    if not bool(bm_cfg_for_overlay.get("save_overlay", False)):
        return

    png_path = bm_dir / "white_bright_mask_overlay.png"
    from .async_plot import submit_task
    submit_task(
        _save_overlay_png,
        cfg, white, mask, objs, matched, input_catalog, image_dict, png_path,
    )
    logger.info("Queued bright-mask overlay: %s", png_path)


def _save_overlay_png(
    cfg: dict[str, Any],
    white: np.ndarray,
    mask: np.ndarray,
    objs: np.ndarray,
    matched: np.ndarray,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, Any],
    png_path: Path,
) -> None:
    """Build and save the full-frame bright-mask overlay diagnostic."""
    try:
        dpi = int(cfg.get("plotting", {}).get("dpi", 150))
        bm_cfg = cfg.get("bright_mask", {})
        match_r = float(bm_cfg.get("match_radius_pix", 3.0))
        ds = max(
            1,
            int(bm_cfg.get("overlay_downsample", cfg.get("overlay", {}).get("downsample_full", 4))),
        )
        h, w = white.shape

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        disp_show = np.asarray(white)[::ds, ::ds].copy()
        m = np.isfinite(disp_show)
        if np.any(m):
            vmin, vmax = ZScaleInterval().get_limits(disp_show[m])
        else:
            vmin, vmax = 0, 1
        disp_show[~m] = 0

        ax.imshow(
            disp_show, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
            interpolation="nearest", rasterized=True, extent=(0, w - 1, 0, h - 1),
        )

        if np.any(mask):
            mask_show = mask[::ds, ::ds]
            mask_rgba = np.zeros((*mask_show.shape, 4), dtype=np.float32)
            mask_rgba[mask_show, 0] = 1.0
            mask_rgba[mask_show, 3] = 0.35
            ax.imshow(
                mask_rgba, origin="lower", interpolation="nearest",
                extent=(0, w - 1, 0, h - 1),
            )

        # -- SEP detections --
        if len(objs) > 0:
            sep_matched_x = [objs["x"][i] for i in range(len(objs)) if matched[i]]
            sep_matched_y = [objs["y"][i] for i in range(len(objs)) if matched[i]]
            sep_unmatched_x = [objs["x"][i] for i in range(len(objs)) if not matched[i]]
            sep_unmatched_y = [objs["y"][i] for i in range(len(objs)) if not matched[i]]
            if sep_matched_x:
                ax.scatter(sep_matched_x, sep_matched_y, marker="o", s=20,
                           facecolors="none", edgecolors="lime", linewidths=0.8,
                           label=f"SEP matched ({len(sep_matched_x)})", zorder=4)
            if sep_unmatched_x:
                ax.scatter(sep_unmatched_x, sep_unmatched_y, marker="x", s=20,
                           c="red", linewidths=0.8,
                           label=f"SEP unmatched ({len(sep_unmatched_x)})", zorder=4)

        # -- Input catalog sources --
        if input_catalog is not None and len(input_catalog) > 0:
            wcs_white = next(iter(image_dict.values()))["wcs"]
            col_map = case_insensitive_col_map(input_catalog.columns)
            ra_col = col_map.get("ra", "RA")
            dec_col = col_map.get("dec", "DEC")

            ra_arr = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
            dec_arr = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
            coord_ok = np.isfinite(ra_arr) & np.isfinite(dec_arr)

            if np.any(coord_ok):
                cx_all = np.full(len(ra_arr), np.nan)
                cy_all = np.full(len(dec_arr), np.nan)
                xw, yw = wcs_white.all_world2pix(ra_arr[coord_ok], dec_arr[coord_ok], 0)
                cx_all[coord_ok] = xw
                cy_all[coord_ok] = yw

                cat_has_sep = np.zeros(len(input_catalog), dtype=bool)
                if len(objs) > 0:
                    det_xy = np.column_stack([
                        np.asarray(objs["x"], dtype=float),
                        np.asarray(objs["y"], dtype=float),
                    ])
                    det_ok = np.isfinite(det_xy).all(axis=1)
                    cat_idx = np.where(
                        coord_ok & np.isfinite(cx_all) & np.isfinite(cy_all)
                    )[0]
                    if np.any(det_ok) and cat_idx.size > 0:
                        tree = cKDTree(det_xy[det_ok])
                        cat_xy = np.column_stack([cx_all[cat_idx], cy_all[cat_idx]])
                        hits = tree.query_ball_point(cat_xy, r=match_r)
                        cat_has_sep[cat_idx] = np.fromiter(
                            (len(h) > 0 for h in hits), dtype=bool, count=len(hits),
                        )

                affected = input_catalog.get("affected_by_bright_mask",
                                             pd.Series(False, index=input_catalog.index)
                                             ).fillna(False).astype(bool).to_numpy()

                cat_matched_m = coord_ok & cat_has_sep
                cat_masked_m = coord_ok & ~cat_has_sep & affected

                if np.any(cat_matched_m):
                    ax.scatter(cx_all[cat_matched_m], cy_all[cat_matched_m],
                               marker="D", s=18, facecolors="none",
                               edgecolors="deepskyblue", linewidths=0.7,
                               label=f"Cat matched ({int(cat_matched_m.sum())})", zorder=3)
                if np.any(cat_masked_m):
                    ax.scatter(cx_all[cat_masked_m], cy_all[cat_masked_m],
                               marker="^", s=22, facecolors="none",
                               edgecolors="orange", linewidths=0.8,
                               label=f"Cat unmatched masked ({int(cat_masked_m.sum())})", zorder=3)

        n_unmatched = int(np.sum(~matched)) if len(matched) > 0 else 0
        n_total = len(objs)
        ax.set_title(f"Bright mask: {n_unmatched} unmatched / {n_total} SEP detections")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        ax.set_xticks([])
        ax.set_yticks([])

        fig.savefig(str(png_path), dpi=dpi)
        plt.close(fig)
    except Exception as e:
        logger.warning("Failed to save bright-mask overlay: %s", e)
