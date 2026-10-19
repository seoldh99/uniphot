"""
White-stack bright unmatched-source masking for tract7dt.

Runs SEP on the in-memory white stack (local sky removed, noise scaled to
the measured scatter), decides which detections are the light of active
catalog sources (light peak within a radius measured in the run, see
``associate_detections``), and masks the others (or hands them to the fit
as subtract-only point sources).  The resulting mask is stored in ``state["bright_mask"]``
and later composed with saturation and non-finite masks into per-band
``bad`` arrays by the pipeline's ``_compose_final_masks`` step, so that
the downstream fit sees ``invvar = 0`` for those pixels.
"""

from __future__ import annotations

import logging
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
    # The threshold is a significance above the local sky in units of the local noise, so both are measured
    # on the white stack itself; the input coadds are used as they are.
    # (1) Sky: some coadds carry an unsubtracted sky pedestal (the UDS coadds have SKYVAL = 0.73-0.84 SKYSIG per
    #     band, +1.6 sigma_white in a 4-band white stack, +4.3 after SEP's 3x3 matched filter, against a 5
    #     sigma threshold: 29-30% of the frame is above it and SEP's active-pixel stack overflows). A SEP mesh
    #     background of the usable pixels is subtracted from this copy.
    # (2) Noise: sigma_white comes from one header SKYSIG per frame. Where a coadd thins out (its footprint
    #     rim) the real scatter is several times larger; the rim then passes the threshold as one snake of
    #     ~1e6 pixels that becomes a frame-sized "bright source" mask. sigma_white is scaled by the locally
    #     measured scatter of (white - sky) / sigma_white (SEP rms on the same mesh), keeping its per-pixel
    #     shape (pixels with fewer usable bands stay noisier).
    # Mesh and filter are SEP's own defaults. Only this detection copy changes: state["white"]/["sigma_white"],
    # the fit data and the saved diagnostics are unchanged; the detections' fluxes (physical mask radius,
    # unmatched-source routing) are then sky-free too.
    det_mask = bad_pix | sig_bad
    bkg = sep.Background(white_sep, mask=det_mask, bw=p["back_size"], bh=p["back_size"],
                         fw=p["back_filter"], fh=p["back_filter"])
    white_sep -= bkg.back(dtype=np.float32)
    white_sep[det_mask] = 0.0
    z = np.zeros_like(white_sep)
    np.divide(white_sep, sig_sep, out=z, where=~det_mask)
    zbkg = sep.Background(z, mask=det_mask, bw=p["back_size"], bh=p["back_size"],
                          fw=p["back_filter"], fh=p["back_filter"])
    scale = zbkg.rms(dtype=np.float32)
    del z
    sig_sep *= scale
    # SEP's rms map is a bicubic spline through the mesh values; next to a sharp step in the scatter (a coadd
    # rim) it can undershoot to zero or below (public example: 28,704 pixels, 0.04% of the frame). A pixel
    # without a positive noise cannot carry a significance: like a pixel without a stated noise, it gets no
    # weight in the detection.
    no_noise = ~det_mask & ~(np.isfinite(sig_sep) & (sig_sep > 0))
    sig_sep[det_mask | no_noise] = 1e30
    logger.info("white-stack detection: SEP mesh %d px, filter %d; sky removed (global %.3g x median sigma_white); "
                "noise scaled by measured scatter (median %.3g, p99 %.3g; not positive at %d px, given no weight)",
                p["back_size"], p["back_filter"],
                float(bkg.globalback) / float(np.median(sigma_white[~det_mask])) if (~det_mask).any() else float("nan"),
                float(np.median(scale[~det_mask])) if (~det_mask).any() else float("nan"),
                float(np.percentile(scale[~det_mask][::97], 99)) if (~det_mask).any() else float("nan"),
                int(no_noise.sum()))
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
        # Sky removed and noise rescaled from the measured scatter before the call (detect_bright_sources):
        # SEP's own default mesh (64 px) and median filter (3 meshes). Part of this dict so a cached objs file
        # made without them is not adopted (the cache key reads this dict).
        "back_size": 64,
        "back_filter": 3,
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

    # The run's PSF, as the monotone radial profile of a representative band: its FWHM sets the association
    # radius below, and in mask_radius_mode "physical" the profile is also the mask radius law.
    psf_profile = _load_white_psf_profile(cfg, state)
    fwhm = _psf_fwhm_from_profile(*psf_profile) if psf_profile is not None else None

    # --- Which detections are the light of active catalog sources ---
    max_shift = float(cfg["compute"]["gpu"]["positions"]["pos_max_shift_pix"])
    matched, assoc = _match_to_catalog(
        objs, input_catalog, image_dict, np.asarray(state["sigma_white"], dtype=np.float32), fwhm, max_shift)
    assoc["fwhm_band"] = _white_psf_band(state) if fwhm is not None else None
    state["bright_mask_matched"] = matched
    state["bright_mask_association"] = assoc
    _log_association(assoc)
    _save_association(cfg, assoc)
    n_matched = int(np.sum(matched))
    n_unmatched = len(objs) - n_matched
    logger.info("Matched %d / %d SEP detections to catalog; %d unmatched to mask",
                n_matched, len(objs), n_unmatched)

    if n_unmatched == 0:
        mask = np.zeros((h, w), dtype=bool)
        input_catalog["affected_by_bright_mask"] = False
        state["bright_mask"] = mask
        _save_diagnostics(cfg, white, mask, objs, matched,
                          input_catalog, image_dict, assoc)
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
                          input_catalog, image_dict, assoc)
        return mask

    # --- Build mask for unmatched detections ---
    # mask_radius_mode: "ellipse" (legacy geometric ellipse_scale/min_radius/
    # dilate) or "physical" (PSF-wing surface brightness vs local noise; no
    # tuned geometric constants -- see _build_physical_mask). Physical falls
    # back to ellipse when no from_stars PSF profile is available.
    mode = str(bm_cfg.get("mask_radius_mode", "ellipse"))
    profile = psf_profile if mode == "physical" else None
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
                      input_catalog, image_dict, assoc)

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


# Association of white-stack detections with the active rows of the catalog.
# A detection is the light of a listed source when an active row lies within r of the detection's matched-filter
# peak pixel (SEP xcpeak, ycpeak: where the light is, not the isophotal centroid, which the shape of the detected
# area -- sky, noise, neighbours -- pulls around). For each population of rows (the user's list; the Gaia rows the
# engine adds as ZP calibrators, which carry their own, Gaia, astrometry)
#     r = min(pos_max_shift_pix, max(FWHM, k * sigma)),   k = sqrt(-2 ln ASSOC_ALPHA)
# - FWHM of the run's PSF: two lights closer than this are one detection; the fit sorts their positions out.
# - sigma, the list-image scatter measured in this run, widens r when the list is worse than the image resolution:
#   with 2-D normal offsets a true pair falls outside k * sigma with probability ASSOC_ALPHA.
# - pos_max_shift_pix: the fit cannot move a source farther from its list position, so light beyond it is not
#   that source's light as far as the fit is concerned; it is masked (or subtracted) as an unlisted source.
# sigma per axis = 1.4826 x median |peak - row| about 0 (a systematic list offset counts as scatter) over the
# mutual nearest row-peak pairs whose detection S/N >= ASSOC_PAIR_MIN_SNR (keeps the peak's own position noise,
# ~0.1 px, below the scatter and chance pairs out); not used with fewer than ASSOC_PAIR_MIN_COUNT pairs (the MAD
# sigma is then uncertain by more than ~15%). sigma = rms of the two axes.
# Detection S/N = isophotal flux / (header sigma_white at the peak pixel x sqrt(npix)).
ASSOC_ALPHA = 1e-3
ASSOC_PAIR_MIN_SNR = 20.0
ASSOC_PAIR_MIN_COUNT = 30
_ASSOC_POPULATIONS = ("listed", "injected_gaia")


def _psf_fwhm_from_profile(psf_r: np.ndarray, psf_p: np.ndarray) -> float | None:
    """FWHM [px] of a monotone radial PSF profile (``psf_radial_profile``): twice the radius where it falls to
    half its central value, linearly interpolated between profile points. None if it never does."""
    r = np.asarray(psf_r, dtype=float)
    p = np.asarray(psf_p, dtype=float)
    if p.size < 2 or not np.isfinite(p[0]) or p[0] <= 0:
        return None
    q = p / p[0]
    below = np.nonzero(q < 0.5)[0]
    if below.size == 0:
        return None
    k = int(below[0])                       # >= 1 because q[0] = 1
    return float(2.0 * (r[k - 1] + (q[k - 1] - 0.5) / (q[k - 1] - q[k]) * (r[k] - r[k - 1])))


def associate_detections(
    objs: np.ndarray,
    row_x: np.ndarray,
    row_y: np.ndarray,
    row_injected: np.ndarray,
    sigma_white: np.ndarray,
    fwhm_pix: float | None,
    max_shift_pix: float,
) -> tuple[np.ndarray, dict]:
    """Pixel-frame association (rule above). ``row_*``: the active rows, ``row_injected`` True for the Gaia rows
    added by the engine. Returns ``matched`` (one bool per detection) and the record of the rule's quantities.
    Without a PSF FWHM (``fwhm_pix`` None) the lower bound of r is ``max_shift_pix``."""
    k = float(np.sqrt(-2.0 * np.log(ASSOC_ALPHA)))
    base = float(max_shift_pix) if fwhm_pix is None else float(fwhm_pix)
    n_det = len(objs)
    row_x = np.asarray(row_x, dtype=float)
    row_y = np.asarray(row_y, dtype=float)
    row_injected = np.asarray(row_injected, dtype=bool)
    ok = np.isfinite(row_x) & np.isfinite(row_y)
    sel = {"listed": ok & ~row_injected, "injected_gaia": ok & row_injected}

    px = np.asarray(objs["xcpeak"], dtype=float) if n_det else np.zeros(0)
    py = np.asarray(objs["ycpeak"], dtype=float) if n_det else np.zeros(0)
    ridx = np.nonzero(ok)[0]
    pair_det = np.full(len(row_x), -1, dtype=np.int64)        # detection index of a usable pair, per row
    if n_det and ridx.size:
        sw = np.asarray(sigma_white, dtype=np.float64)
        h, w = sw.shape
        ix = np.clip(px.astype(np.int64), 0, w - 1)
        iy = np.clip(py.astype(np.int64), 0, h - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = np.asarray(objs["flux"], dtype=float) / (sw[iy, ix] * np.sqrt(np.asarray(objs["npix"], float)))
        peak_xy = np.column_stack([px, py])
        row_xy = np.column_stack([row_x[ridx], row_y[ridx]])
        _, nd = cKDTree(peak_xy).query(row_xy)             # nearest peak of each row
        _, nr = cKDTree(row_xy).query(peak_xy)             # nearest row of each peak
        use = (nr[nd] == np.arange(ridx.size)) & (snr[nd] >= ASSOC_PAIR_MIN_SNR)
        pair_det[ridx[use]] = nd[use]

    pops: dict[str, dict] = {}
    for name in _ASSOC_POPULATIONS:
        s = sel[name] & (pair_det >= 0)
        n_pairs = int(s.sum())
        rec = {"n_rows": int(sel[name].sum()), "n_pairs": n_pairs, "scatter_measured": False,
               "sigma_x_pix": None, "sigma_y_pix": None, "sigma_pix": None}
        r = min(float(max_shift_pix), base)
        if n_pairs >= ASSOC_PAIR_MIN_COUNT:
            dx = px[pair_det[s]] - row_x[s]
            dy = py[pair_det[s]] - row_y[s]
            sx = 1.4826 * float(np.median(np.abs(dx)))
            sy = 1.4826 * float(np.median(np.abs(dy)))
            sp = float(np.sqrt(0.5 * (sx * sx + sy * sy)))
            rec.update(scatter_measured=True, sigma_x_pix=sx, sigma_y_pix=sy, sigma_pix=sp)
            r = min(float(max_shift_pix), max(base, k * sp))
        rec["radius_pix"] = float(r)
        pops[name] = rec

    matched = np.zeros(n_det, dtype=bool)
    if n_det:
        peak_xy = np.column_stack([px, py])
        for name in _ASSOC_POPULATIONS:
            if not sel[name].any():
                continue
            d, _ = cKDTree(np.column_stack([row_x[sel[name]], row_y[sel[name]]])).query(peak_xy)
            matched |= d <= pops[name]["radius_pix"]
    record = {
        "position": "matched-filter peak pixel (SEP xcpeak, ycpeak)",
        "fwhm_pix": None if fwhm_pix is None else float(fwhm_pix),
        "max_shift_pix": float(max_shift_pix),
        "alpha": ASSOC_ALPHA, "k": k,
        "pair_min_snr": ASSOC_PAIR_MIN_SNR, "pair_min_count": ASSOC_PAIR_MIN_COUNT,
        "populations": pops,
        "n_detections": int(n_det), "n_matched": int(matched.sum()), "n_unmatched": int(n_det - matched.sum()),
    }
    return matched, record


def _catalog_pixels(catalog: pd.DataFrame, image_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """White-stack pixel positions of the catalog rows (NaN where RA/Dec is not finite)."""
    wcs_white = next(iter(image_dict.values()))["wcs"]
    col_map = case_insensitive_col_map(catalog.columns)
    ra = pd.to_numeric(catalog[col_map.get("ra", "RA")], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(catalog[col_map.get("dec", "DEC")], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    x = np.full(len(ra), np.nan, dtype=float)
    y = np.full(len(ra), np.nan, dtype=float)
    if np.any(ok):
        x[ok], y[ok] = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
    return x, y


def _is_injected(catalog: pd.DataFrame) -> np.ndarray:
    if "_is_injected_gaia" not in catalog.columns:
        return np.zeros(len(catalog), dtype=bool)
    return catalog["_is_injected_gaia"].fillna(False).astype(bool).to_numpy()


def _match_to_catalog(
    objs: np.ndarray,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, Any],
    sigma_white: np.ndarray,
    fwhm_pix: float | None,
    max_shift_pix: float,
) -> tuple[np.ndarray, dict]:
    """:func:`associate_detections` for the active rows of ``input_catalog`` (not ``excluded_any``)."""
    is_active = ~input_catalog.get("excluded_any", pd.Series(False, index=input_catalog.index)).fillna(False).astype(bool)
    active_cat = input_catalog.loc[is_active]
    if len(active_cat) == 0:
        x = y = np.zeros(0)
        inj = np.zeros(0, dtype=bool)
    else:
        x, y = _catalog_pixels(active_cat, image_dict)
        inj = _is_injected(active_cat)
    return associate_detections(objs, x, y, inj, sigma_white, fwhm_pix, max_shift_pix)


def _row_radii(catalog: pd.DataFrame, assoc: dict) -> np.ndarray:
    """Association radius of each catalog row (by its population)."""
    pops = assoc["populations"]
    return np.where(_is_injected(catalog), pops["injected_gaia"]["radius_pix"], pops["listed"]["radius_pix"])


def _log_association(assoc: dict) -> None:
    def pop(name, label):
        p = assoc["populations"][name]
        sc = (f"scatter {p['sigma_pix']:.2f} px ({p['sigma_x_pix']:.2f}, {p['sigma_y_pix']:.2f}) from "
              f"{p['n_pairs']} pairs" if p["scatter_measured"]
              else f"scatter not measured ({p['n_pairs']} pairs < {ASSOC_PAIR_MIN_COUNT})")
        return f"{label} {p['n_rows']}: {sc}, r {p['radius_pix']:.2f} px"

    fw = assoc["fwhm_pix"]
    if fw is None:
        logger.warning("bright-mask association: no PSF profile, so no FWHM; the radius floor is "
                       "pos_max_shift_pix = %.2f px", assoc["max_shift_pix"])
    logger.info(
        "bright-mask association (peak within r = min(%.2f, max(FWHM, %.2f x scatter)) of an active row): "
        "FWHM %s; %s; %s; %d / %d detections are the light of active rows",
        assoc["max_shift_pix"], assoc["k"],
        "n/a" if fw is None else f"{fw:.2f} px (band {assoc.get('fwhm_band')})",
        pop("listed", "listed rows"), pop("injected_gaia", "added Gaia rows"),
        assoc["n_matched"], assoc["n_detections"])


def _save_association(cfg: dict[str, Any], assoc: dict) -> None:
    import json

    wd = (cfg.get("outputs", {}) or {}).get("work_dir")
    if not wd:
        return
    p = Path(wd) / "bright_mask" / "association.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(assoc, indent=1) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("bright-mask association record not written (%s)", exc)


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
    unit-flux radial profile ``(psf_r, psf_p)`` for the physical-radius mask
    and for the FWHM of the detection-catalogue association.

    Returns ``None`` when no from_stars PSF is available (the PSF stage did
    not run, or its file cannot be read): the mask then falls back to the
    ellipse geometry and the association radius floor to pos_max_shift_pix. The PSF stage
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

        band = _white_psf_band(state)
        if band is None:
            return None
        white = np.asarray(state["white"])
        h, w = white.shape
        psf = load_psf_polynomial_for_band(_Path(str(psf_dir)), band)
        stamp = evaluate_psf_at_np(psf, w / 2.0, h / 2.0)
        psf_r, psf_p = psf_radial_profile(np.asarray(stamp, dtype=float))
        return psf_r, psf_p
    except Exception as e:  # pragma: no cover - defensive fallback
        logger.warning("bright mask: could not load the white-stack PSF profile (%s)", e)
        return None


def _white_psf_band(state: dict[str, Any]) -> str | None:
    """The band whose PSF stands for the white stack: the middle one of ``image_dict``."""
    bands = list(state["image_dict"].keys())
    return bands[len(bands) // 2] if bands else None


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
    assoc: dict | None = None,
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
        cfg, white, mask, objs, matched, input_catalog, image_dict, png_path, assoc,
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
    assoc: dict | None = None,
) -> None:
    """Build and save the full-frame bright-mask overlay diagnostic. A catalog row
    counts as "matched" when a detection peak lies within its association radius."""
    try:
        dpi = int(cfg.get("plotting", {}).get("dpi", 150))
        bm_cfg = cfg.get("bright_mask", {})
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
                if len(objs) > 0 and assoc is not None:
                    peak_xy = np.column_stack([
                        np.asarray(objs["xcpeak"], dtype=float),
                        np.asarray(objs["ycpeak"], dtype=float),
                    ])
                    cat_idx = np.where(
                        coord_ok & np.isfinite(cx_all) & np.isfinite(cy_all)
                    )[0]
                    if cat_idx.size > 0:
                        cat_xy = np.column_stack([cx_all[cat_idx], cy_all[cat_idx]])
                        d_peak, _ = cKDTree(peak_xy).query(cat_xy)
                        cat_has_sep[cat_idx] = d_peak <= _row_radii(input_catalog, assoc)[cat_idx]

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
