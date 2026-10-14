"""
Zero-point calibration using GaiaXP synthetic photometry.

Two entry points:
  - augment_catalog_with_gaia(): inject Gaia synphot sources into the input catalog
  - compute_zp(): derive per-band ZP from fitted Gaia stars, apply AB magnitudes
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u

from .gaia_astrometry import (
    GAIA_POS_ERR_COLS,
    estimate_frame_tie_floor_mas,
    obs_epoch_from_headers,
    propagate_gaia_pos_err,
    write_frame_tie_floor,
)
from .fit.grid_gpu.band_flags import FLAG_SATURATED, FLAG_ZP_EXTRAP
from . import run_status
from .run_status import RunContractError
from .utils import case_insensitive_col_map, write_csv_parallel

logger = logging.getLogger("tract7dt.zp")

K_MAG = 2.5 / np.log(10.0)

# How far the spatial ZP model may stray from what the calibrators around
# a position say before the model counts as unsupported there. 0.1 mag is the
# size the development plan treats as a severe defect (a silently wrong value of
# 0.1 mag or more). Smaller disagreements are the model's general underfit: on the
# two development test fields (a development analysis script, not distributed) the
# model differs from the median of its 30 nearest calibrators by more than the
# reported ZP error in 2-30% of the field, by more than 0.1 mag in <= 0.15%.
ZP_SUPPORT_TOL_MAG = 0.1

# Background Gaia-quality fetch of this run (set by augment_catalog_with_gaia,
# awaited by require_gaia_quality_flags before the fit): thread, candidate
# ids, cache path, deadline.
_GAIA_QF_PREFETCH: dict[str, Any] = {}


# ---------------------------------------------------------------------------
#  Gaia catalog augmentation
# ---------------------------------------------------------------------------

def gaia_in_image(x: np.ndarray, y: np.ndarray, *, img_h: int, img_w: int) -> np.ndarray:
    """R28: the Gaia calibrator candidates are every source whose position
    falls on the (cropped) image -- a property of the image, never of the
    input catalog. The old selection box was built around the catalog, so in
    a few-target run the outermost targets always sat on the calibrators'
    hull edge (R08 bit 4 in almost every band) and the ZP depended on which
    targets were run together."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return (np.isfinite(x) & np.isfinite(y)
            & (x >= 0) & (x < img_w) & (y >= 0) & (y < img_h))


def _save_augmentation_overlay(
    *,
    white: np.ndarray,
    wcs: Any,
    x_orig: np.ndarray,
    y_orig: np.ndarray,
    x_gaia_new: np.ndarray,
    y_gaia_new: np.ndarray,
    x_gaia_matched: np.ndarray,
    y_gaia_matched: np.ndarray,
    x_gaia_sat: np.ndarray,
    y_gaia_sat: np.ndarray,
    excluded_crop: np.ndarray | None = None,
    excluded_saturation: np.ndarray | None = None,
    outpath: Path,
    dpi: int = 150,
) -> None:
    """Overlay original + injected Gaia sources on the white image."""
    from astropy.visualization import ZScaleInterval

    outpath.parent.mkdir(parents=True, exist_ok=True)

    H, W = white.shape
    interval = ZScaleInterval()
    finite = white[np.isfinite(white)]
    vmin, vmax = interval.get_limits(finite) if len(finite) > 0 else (0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
    ax.imshow(white, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    ok_o = np.isfinite(x_orig) & np.isfinite(y_orig)
    if np.any(ok_o):
        ax.scatter(x_orig[ok_o], y_orig[ok_o], s=20, marker="o", facecolors="none",
                   edgecolors="cyan", linewidths=0.8, alpha=0.8, label=f"original ({int(ok_o.sum())})")

    ok_m = np.isfinite(x_gaia_matched) & np.isfinite(y_gaia_matched)
    if np.any(ok_m):
        ax.scatter(x_gaia_matched[ok_m], y_gaia_matched[ok_m], s=30, marker="s", facecolors="none",
                   edgecolors="lime", linewidths=0.8, alpha=0.8, label=f"Gaia matched ({int(ok_m.sum())})")

    ok_n = np.isfinite(x_gaia_new) & np.isfinite(y_gaia_new)
    if np.any(ok_n):
        ax.scatter(x_gaia_new[ok_n], y_gaia_new[ok_n], s=20, marker="o", facecolors="none",
                   edgecolors="magenta", linewidths=0.8, alpha=0.8, label=f"Gaia injected ({int(ok_n.sum())})")

    ok_s = np.isfinite(x_gaia_sat) & np.isfinite(y_gaia_sat)
    if np.any(ok_s):
        ax.scatter(x_gaia_sat[ok_s], y_gaia_sat[ok_s], s=40, marker="x",
                   c="red", linewidths=1.0, alpha=0.9, label=f"Gaia saturated ({int(ok_s.sum())})")

    if excluded_saturation is not None:
        sat_mask = excluded_saturation & ok_o
        n_sat = int(sat_mask.sum())
        if n_sat > 0:
            ax.scatter(x_orig[sat_mask], y_orig[sat_mask], s=60, marker="x",
                       c="red", linewidths=1.2, alpha=0.9, zorder=5,
                       label=f"excluded: saturation ({n_sat})")

    if excluded_crop is not None:
        crop_mask = excluded_crop & ok_o
        n_crop = int(crop_mask.sum())
        if n_crop > 0:
            ax.scatter(x_orig[crop_mask], y_orig[crop_mask], s=60, marker="x",
                       c="orange", linewidths=1.2, alpha=0.9, zorder=5,
                       label=f"excluded: crop ({n_crop})")

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(-0.5, H - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_title("Gaia augmentation overlay on white stack")
    from .async_plot import submit_plot
    submit_plot(fig, outpath, dpi=dpi, bbox_inches="tight")
    logger.info("Queued Gaia augmentation overlay: %s", outpath)


def saturated_in_any_band(cat: pd.DataFrame) -> np.ndarray:
    """True for rows whose ``FLAGS_{band}_fit`` carries the saturation bit (8)
    in any band (R11). All False for catalogs without band flags."""
    sat = np.zeros(len(cat), dtype=bool)
    for c in cat.columns:
        if c.startswith("FLAGS_") and c.endswith("_fit"):
            f = pd.to_numeric(cat[c], errors="coerce").fillna(0).astype(int)
            sat |= (f.to_numpy() & FLAG_SATURATED) > 0
    return sat


def _filter_gaia_saturation(
    gaia_df: pd.DataFrame,
    *,
    image_dict: dict,
    wcs: Any,
    radius_pix: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Per-band saturation cut for Gaia sources to inject (R11), mirroring the
    catalog cut: a source is dropped only when a saturated pixel lies within
    ``radius_pix`` in EVERY band; the kept sources carry their per-band hits
    (returned as ``{band: bool array}`` aligned with the kept frame) so the
    affected bands are excluded per band downstream."""
    from .load_inputs import saturation_hits_per_band
    ra = gaia_df["ra"].to_numpy(dtype=float)
    dec = gaia_df["dec"].to_numpy(dtype=float)
    x, y = wcs.all_world2pix(ra, dec, 0)
    ix = np.rint(x).astype(int)
    iy = np.rint(y).astype(int)
    finite = np.isfinite(x) & np.isfinite(y)

    hits = saturation_hits_per_band(image_dict, ix, iy, finite, radius_pix)
    sat_hit = finite.copy() if hits else np.zeros(len(gaia_df), dtype=bool)
    for h in hits.values():
        sat_hit &= h
    n_removed = int(sat_hit.sum())
    if n_removed > 0:
        logger.info("Saturation-cut removed %d/%d injected Gaia sources "
                    "(saturated in every band)", n_removed, len(gaia_df))
    sat_x = x[sat_hit] if n_removed > 0 else np.array([])
    sat_y = y[sat_hit] if n_removed > 0 else np.array([])
    kept = gaia_df.loc[~sat_hit].copy().reset_index(drop=True)
    kept_hits = {b: np.asarray(h)[~sat_hit] for b, h in hits.items()}
    return kept, sat_x, sat_y, kept_hits


def _resolve_injected_pos_err(
    new_gaia: pd.DataFrame,
    *,
    mode: str,
    fixed_pix: float | None,
    epoch_obs_yr: float | None,
    plate_scale_arcsec_px: float | None,
    floor_mas: float = 0.0,
) -> tuple[np.ndarray | None, str]:
    """POS_ERR values [px] for the pipeline-injected Gaia calibrator rows.

    Returns ``(values, effective_mode)``:

    * mode ``"fixed"``: the legacy constant ``fixed_pix`` on every injected
      row (``(None, "none")`` when ``fixed_pix`` is null = no prior).
    * mode ``"propagate"``: per-source widths epoch-propagated from the Gaia
      DR3 astrometric errors (:func:`~tract7dt.gaia_astrometry.propagate_gaia_pos_err`);
      rows without a PM solution get NaN (= no prior, deliberate). Falls back
      to the ``"fixed"`` behavior (logged) when the GaiaXP CSV lacks the
      astrometric-error columns or no observation epoch / plate scale could
      be derived from the images.
    """
    if mode not in ("fixed", "propagate"):
        raise ValueError(
            f"zp.gaia_pos_err_mode must be 'fixed' or 'propagate', got {mode!r}")
    if mode == "propagate":
        missing = [c for c in GAIA_POS_ERR_COLS if c not in new_gaia.columns]
        if missing:
            logger.warning(
                "zp.gaia_pos_err_mode='propagate' requested but the GaiaXP CSV "
                "lacks the Gaia astrometric-error column(s) %s; falling back "
                "to the fixed width (zp.gaia_pos_err_pix=%s). Add ra_error/"
                "dec_error/pmra_error/pmdec_error (+ optionally "
                "astrometric_params_solved) to enable per-source widths.",
                missing, fixed_pix)
        elif epoch_obs_yr is None or not np.isfinite(epoch_obs_yr):
            logger.warning(
                "zp.gaia_pos_err_mode='propagate': no observation epoch could "
                "be derived from the image headers (MJD-OBS/DATE-OBS); falling "
                "back to the fixed width (zp.gaia_pos_err_pix=%s).", fixed_pix)
        elif (plate_scale_arcsec_px is None
              or not np.isfinite(plate_scale_arcsec_px)
              or plate_scale_arcsec_px <= 0):
            logger.warning(
                "zp.gaia_pos_err_mode='propagate': no plate scale could be "
                "derived from the image WCS; falling back to the fixed width "
                "(zp.gaia_pos_err_pix=%s).", fixed_pix)
        else:
            vals = propagate_gaia_pos_err(
                new_gaia, float(epoch_obs_yr), float(plate_scale_arcsec_px),
                floor_mas=float(floor_mas))
            return vals, "propagate"
    if fixed_pix is None:
        return None, "none"
    return np.full(len(new_gaia), float(fixed_pix), dtype=float), "fixed"


def augment_catalog_with_gaia(
    *,
    cfg: dict,
    state: dict,
) -> pd.DataFrame:
    """Inject Gaia synphot sources into the input catalog for ZP calibration.

    Returns the augmented catalog (also saved to disk). Updates state in-place.
    """
    zp_cfg = cfg["zp"]
    outputs = cfg["outputs"]
    inputs = cfg["inputs"]
    zp_ref = float(cfg["image_scaling"]["zp_ref"])

    gaia_csv_path = inputs.get("gaiaxp_synphot_csv")
    if gaia_csv_path is None or not Path(gaia_csv_path).exists():
        logger.warning("GaiaXP synphot CSV not found (%s); skipping augmentation", gaia_csv_path)
        return state["input_catalog"]

    input_catalog: pd.DataFrame = state["input_catalog"]
    image_dict: dict = state["image_dict"]
    white: np.ndarray = state["white"]

    gaia_mag_min = float(zp_cfg.get("gaia_mag_min", 11.0))
    gaia_mag_max = float(zp_cfg.get("gaia_mag_max", 18.0))
    match_radius = float(zp_cfg.get("match_radius_arcsec", 1.0))

    gaia = pd.read_csv(gaia_csv_path, dtype={"source_id": str})
    logger.info("Loaded GaiaXP synphot: %d sources", len(gaia))
    # Cache the raw synphot frame so compute_zp can reuse it instead of re-reading
    # the same file (the frame is not mutated below; gaia_filt is a .copy()).
    state["_gaia_synphot_df"] = gaia

    for c in ("ra", "dec", "source_id", "phot_g_mean_mag"):
        if c not in gaia.columns:
            raise ValueError(f"GaiaXP CSV missing required column: {c}")

    gaia_filt = gaia[
        (gaia["phot_g_mean_mag"] >= gaia_mag_min)
        & (gaia["phot_g_mean_mag"] <= gaia_mag_max)
    ].copy()
    logger.info("After mag cut [%.1f, %.1f]: %d sources", gaia_mag_min, gaia_mag_max, len(gaia_filt))

    wcs = next(iter(image_dict.values()))["wcs"]
    H, W = white.shape

    col_map = case_insensitive_col_map(input_catalog.columns)
    ra_col = col_map.get("ra", "RA")
    dec_col = col_map.get("dec", "DEC")
    id_col = col_map.get("id", "ID") if "id" in col_map else None

    cat_ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
    cat_dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)

    x_cat, y_cat = wcs.all_world2pix(cat_ra, cat_dec, 0)

    gaia_ra = gaia_filt["ra"].to_numpy(dtype=float)
    gaia_dec = gaia_filt["dec"].to_numpy(dtype=float)
    x_gaia, y_gaia = wcs.all_world2pix(gaia_ra, gaia_dec, 0)

    in_img = gaia_in_image(x_gaia, y_gaia, img_h=H, img_w=W)
    gaia_img = gaia_filt.loc[in_img].copy().reset_index(drop=True)
    logger.info("Gaia sources in the image (%dx%d px, independent of the input "
                "catalog): %d", W, H, len(gaia_img))

    cat_sc = SkyCoord(ra=cat_ra * u.deg, dec=cat_dec * u.deg)
    gaia_sc = SkyCoord(ra=gaia_img["ra"].to_numpy() * u.deg, dec=gaia_img["dec"].to_numpy() * u.deg)

    if len(gaia_sc) == 0 or len(cat_sc) == 0:
        logger.warning("No Gaia sources in the image or empty input catalog; skipping augmentation")
        input_catalog = input_catalog.copy()
        input_catalog["gaia_source_id"] = ""
        state["input_catalog"] = input_catalog
        return input_catalog

    idx_gaia, sep2d, _ = cat_sc.match_to_catalog_sky(gaia_sc)
    sep_arcsec = sep2d.to(u.arcsec).value
    matched_mask = sep_arcsec <= match_radius

    aug_cat = input_catalog.copy().reset_index(drop=True)
    aug_cat["_is_injected_gaia"] = False
    gaia_sid_arr = np.full(len(aug_cat), "", dtype=object)
    if np.any(matched_mask):
        gaia_source_ids = gaia_img["source_id"].astype(str).to_numpy()
        gaia_sid_arr[matched_mask] = gaia_source_ids[idx_gaia[matched_mask]]
    aug_cat["gaia_source_id"] = gaia_sid_arr

    bandnames = sorted(image_dict.keys())

    n_filled = 0
    matched_indices = np.where(matched_mask)[0]
    matched_gaia_indices = idx_gaia[matched_indices] if len(matched_indices) else np.array([], dtype=int)
    for bn in bandnames:
        flux_col = f"FLUX_{bn}"
        mag_col = f"mag_{bn}"
        if flux_col not in aug_cat.columns or mag_col not in gaia_img.columns:
            continue
        val = pd.to_numeric(aug_cat.loc[matched_indices, flux_col], errors="coerce").to_numpy(dtype=float)
        needs_fill = pd.isna(val) | (val == 0)
        if not np.any(needs_fill):
            continue
        mags = gaia_img[mag_col].to_numpy(dtype=float)[matched_gaia_indices]
        ok_mag = np.isfinite(mags) & needs_fill
        if not np.any(ok_mag):
            continue
        fill_idx = matched_indices[ok_mag]
        fill_vals = 10.0 ** ((zp_ref - mags[ok_mag]) / 2.5)
        aug_cat.loc[fill_idx, flux_col] = fill_vals
        n_filled += int(len(fill_idx))
    if n_filled > 0:
        logger.info("Backfilled %d missing FLUX values for matched sources from Gaia synphot", n_filled)

    inject_gaia = bool(zp_cfg.get("inject_gaia_sources", True))
    gaia_pos_err_pix = zp_cfg.get("gaia_pos_err_pix", None)
    # ".get default 'fixed'" keeps hand-built cfg dicts on the exact legacy
    # behavior; configs loaded through load_config carry the DEFAULT_CONFIG
    # default ("propagate", with automatic fixed-width fallback).
    gaia_pos_err_mode = str(
        zp_cfg.get("gaia_pos_err_mode", "fixed")).strip().lower()

    matched_gaia_ids = set(v for v in gaia_sid_arr if v)
    new_gaia = gaia_img[~gaia_img["source_id"].isin(matched_gaia_ids)].copy().reset_index(drop=True)
    from .load_inputs import SATURATION_BAND_COLUMN_PREFIX
    x_gaia_sat = np.array([])
    y_gaia_sat = np.array([])
    gaia_sat_hits: dict = {}
    new_rows: list[dict] = []

    if inject_gaia:
        sat_cut_cfg = cfg.get("source_saturation_cut", {})
        if bool(sat_cut_cfg.get("enabled", True)) and len(new_gaia) > 0:
            sat_radius = float(sat_cut_cfg.get("radius_pix", 20.0))
            new_gaia, x_gaia_sat, y_gaia_sat, gaia_sat_hits = \
                _filter_gaia_saturation(
                    new_gaia, image_dict=image_dict, wcs=wcs,
                    radius_pix=sat_radius,
                )

        logger.info("Original sources matched to Gaia: %d | New Gaia sources to inject: %d",
                    int(matched_mask.sum()), len(new_gaia))

        pos_err_epoch_yr: float | None = None
        pos_err_plate_scale: float | None = None
        if gaia_pos_err_mode == "propagate":
            pos_err_epoch_yr = obs_epoch_from_headers(
                d.get("hdr") for d in image_dict.values())
            try:
                from astropy.wcs.utils import proj_plane_pixel_scales
                pos_err_plate_scale = float(
                    np.mean(proj_plane_pixel_scales(wcs)) * 3600.0)
            except Exception:
                pos_err_plate_scale = None
        pos_err_vals, pos_err_mode_eff = _resolve_injected_pos_err(
            new_gaia,
            mode=gaia_pos_err_mode,
            fixed_pix=gaia_pos_err_pix,
            epoch_obs_yr=pos_err_epoch_yr,
            plate_scale_arcsec_px=pos_err_plate_scale,
            floor_mas=float(zp_cfg.get("gaia_pos_err_floor_mas", 0.0) or 0.0),
        )

        # new_gaia is reset_index(drop=True) above (and by the saturation
        # filter), so the iterrows index is positional.
        for gi, g in new_gaia.iterrows():
            row: dict[str, Any] = {}
            if id_col:
                row[id_col] = f"gaia_{g['source_id']}"
            row[ra_col] = float(g["ra"])
            row[dec_col] = float(g["dec"])
            type_col = col_map.get("type", "TYPE")
            row[type_col] = "STAR"
            row["gaia_source_id"] = str(g["source_id"])
            row["_is_injected_gaia"] = True
            for _b, _h in gaia_sat_hits.items():
                row[f"{SATURATION_BAND_COLUMN_PREFIX}{_b}"] = bool(_h[gi])
            if pos_err_vals is not None:
                # NaN (propagate mode, no PM solution) = deliberate no-prior.
                row["POS_ERR"] = float(pos_err_vals[gi])

            for bn in bandnames:
                mag_col = f"mag_{bn}"
                flux_col_name = f"FLUX_{bn}"
                if mag_col in g.index and np.isfinite(g[mag_col]):
                    row[flux_col_name] = float(10.0 ** ((zp_ref - float(g[mag_col])) / 2.5))
                else:
                    row[flux_col_name] = 0.0
            new_rows.append(row)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            aug_cat = pd.concat([aug_cat, new_df], ignore_index=True)
            if pos_err_vals is not None and pos_err_mode_eff == "propagate":
                n_prop = int(np.isfinite(pos_err_vals).sum())
                med_px = float(np.nanmedian(pos_err_vals)) if n_prop else float("nan")
                logger.info(
                    "Assigned epoch-propagated Gaia POS_ERR to %d/%d injected "
                    "Gaia sources (median %.4f px at epoch %.2f; %d without a "
                    "PM solution carry NaN = no prior)",
                    n_prop, len(new_rows), med_px,
                    float(pos_err_epoch_yr), len(new_rows) - n_prop,
                )
            elif pos_err_vals is not None:
                logger.info(
                    "Assigned POS_ERR=%.3f to %d injected Gaia sources",
                    float(gaia_pos_err_pix),
                    len(new_rows),
                )
    else:
        logger.info("Original sources matched to Gaia: %d | Gaia injection disabled (inject_gaia_sources=false); "
                    "%d unmatched Gaia sources skipped", int(matched_mask.sum()), len(new_gaia))

    zp_dir = Path(outputs["work_dir"]) / "ZP"
    zp_dir.mkdir(parents=True, exist_ok=True)

    out_name = Path(inputs["input_catalog"]).stem + "_with_Gaia.csv"
    out_path = zp_dir / out_name
    # The GPU fit consumes the augmented catalog from state['input_catalog'] in
    # memory (set below); no CSV is written (the per-patch flow that read it
    # was removed in card 018).
    logger.info("Skipping augmented-catalog CSV write in gpu mode "
                "(no downstream reader): %s", out_path)

    if new_rows:
        new_ra = np.array([r[ra_col] for r in new_rows], dtype=float)
        new_dec = np.array([r[dec_col] for r in new_rows], dtype=float)
        x_new_gaia, y_new_gaia = wcs.all_world2pix(new_ra, new_dec, 0)
    else:
        x_new_gaia, y_new_gaia = np.array([]), np.array([])

    matched_orig_indices = np.where(matched_mask)[0]
    x_matched = x_cat[matched_orig_indices] if len(matched_orig_indices) else np.array([])
    y_matched = y_cat[matched_orig_indices] if len(matched_orig_indices) else np.array([])

    _excl_crop = input_catalog["excluded_crop"].fillna(False).to_numpy(dtype=bool) if "excluded_crop" in input_catalog.columns else None
    _excl_sat = input_catalog["excluded_saturation"].fillna(False).to_numpy(dtype=bool) if "excluded_saturation" in input_catalog.columns else None

    if bool(cfg.get("zp", {}).get("save_gaia_overlay", False)):
        plot_dpi = int(cfg.get("plotting", {}).get("dpi", 150))
        _save_augmentation_overlay(
            white=white, wcs=wcs,
            x_orig=x_cat, y_orig=y_cat,
            x_gaia_new=x_new_gaia, y_gaia_new=y_new_gaia,
            x_gaia_matched=x_matched, y_gaia_matched=y_matched,
            x_gaia_sat=x_gaia_sat, y_gaia_sat=y_gaia_sat,
            excluded_crop=_excl_crop,
            excluded_saturation=_excl_sat,
            outpath=zp_dir / "gaia_augmentation_overlay.png",
            dpi=plot_dpi,
        )

    # --- Image<->Gaia frame-tie: NO pre-fit POS_ERR modification (C2 REVERTED
    # 2026-07-11). POS_ERR stays the pure epoch-propagated Gaia width (a KNOWN
    # per-source input uncertainty). The coherent image<->Gaia offset is a
    # FIELD-LEVEL / WCS systematic, not a per-source error bar, and must not be
    # back-measured against the Gaia seed and folded into POS_ERR (it is also
    # undefined for the reference-less science sources such a floor would then
    # touch). The `zp.frame_tie_diagnostic: "measure"` knob may LOG + persist the
    # offset for the record in compute_zp, but it never feeds POS_ERR.

    state["input_catalog"] = aug_cat
    state["augmented_catalog_path"] = out_path

    # Fetch the Gaia DR3 quality flags (RUWE + BP/RP excess) of every possible
    # calibrator in the background while the PSF stage runs; the ZP filters
    # its calibrators on them. Every possible calibrator is a row of aug_cat
    # with a non-empty gaia_source_id, so this superset covers compute_zp.
    # R29: require_gaia_quality_flags() waits for it before the fit, inside
    # one time budget, and fails the run if the flags cannot be obtained.
    _GAIA_QF_PREFETCH.clear()
    flag_cols = {"ruwe", "phot_bp_rp_excess_factor"}
    if (bool(zp_cfg.get("gaia_quality_filter", True))
            and not flag_cols <= set(gaia.columns)):
        prefetch_ids = [
            s for s in aug_cat["gaia_source_id"].astype(str).str.strip().unique()
            if s and s.lower() != "nan"
        ]
        cache_path = zp_dir / "gaia_quality_cache.csv"

        from .gaia_quality import QUALITY_BUDGET_S, prefetch_quality_flags_async

        # No fetch-parameter overrides here, ON PURPOSE: the prefetch must be
        # the SAME call as the verification fetch in require_gaia_quality_flags
        # (same deadline). See prefetch_quality_flags_async for the
        # measurement that retired the old `max_workers=1` pin ("speed does not
        # matter here" was false: 1 worker projected ~33 min on a crowded tile
        # where 8 workers took 3m46s on the same ids).
        deadline = time.monotonic() + QUALITY_BUDGET_S
        _t = prefetch_quality_flags_async(
            prefetch_ids,
            cache_path=str(cache_path),
            deadline=deadline,
            on_error=lambda exc: logger.info(
                "Gaia quality prefetch failed (%s); checked again before the "
                "fit", exc),
        )
        _GAIA_QF_PREFETCH.update(thread=_t, ids=prefetch_ids,
                                 cache_path=str(cache_path), deadline=deadline)
        logger.info("Gaia quality: background prefetch of RUWE + BP/RP-excess "
                    "started for %d candidate calibrators", len(prefetch_ids))

    return aug_cat


def require_gaia_quality_flags(cfg: dict, state: dict | None = None) -> None:
    """R29: obtain the Gaia quality flags of every calibrator candidate before
    the fit, or fail the run there.

    With ``zp.gaia_quality_filter`` on and a GaiaXP CSV without the flag
    columns, the Gaia augment stage started a background fetch into
    ``<work_dir>/ZP/gaia_quality_cache.csv``. This waits for it within the
    run's one time budget (``gaia_quality.QUALITY_BUDGET_S`` from the fetch
    start), fetches any remainder inside the same budget, and raises
    ``RunContractError("gaia_quality_unavailable")`` if the cache still does
    not cover every candidate. It used to be "graceful": a failed fetch left
    the ZP calibrated on the unfiltered set (crowded T08285 2026-09-27:
    calibrators +23-42%, bit-4 cells +-hundreds per band) while RUN_STATUS
    said "ok" -- the same tile got a different ZP depending on the network.
    Failing before the fit costs at most the stages before it.
    """
    zp_cfg = cfg.get("zp", {}) or {}
    if not (bool(zp_cfg.get("enabled", False))
            and bool(zp_cfg.get("gaia_quality_filter", True))):
        return
    if "ids" not in _GAIA_QF_PREFETCH:
        # The GaiaXP CSV carries the flags (or there is no GaiaXP CSV, which
        # compute_zp reports as its own failure): nothing to obtain.
        return
    from .gaia_quality import QUALITY_BUDGET_S, load_quality_flags
    pre = _GAIA_QF_PREFETCH
    t, ids, deadline = pre.get("thread"), pre["ids"], float(pre["deadline"])
    t0 = time.monotonic()
    if t is not None and t.is_alive():
        logger.info("Gaia quality: waiting for the background fetch before "
                    "the fit (%.0f s of the %.0f s budget left)...",
                    max(0.0, deadline - t0), QUALITY_BUDGET_S)
        # The fetch itself stops at the deadline; the extra seconds only let
        # an in-flight request return.
        t.join(timeout=max(0.0, deadline - t0) + 30.0)
    if t is not None and t.is_alive():
        t.stop.set()
        raise RunContractError(
            "gaia_quality_unavailable",
            f"the Gaia quality flags of {len(ids)} calibrator candidates were "
            f"not obtained within {QUALITY_BUDGET_S:.0f} s; the run stops "
            "before the fit instead of calibrating the ZP on the unfiltered "
            "set. Re-run when the Gaia archive answers, pre-fetch with "
            "`tract7dt augment-gaia <csv>`, or set zp.gaia_quality_filter: "
            "false", n_candidates=len(ids))
    prefetch_error = getattr(t, "error", None)
    try:
        flags = load_quality_flags(ids, cache_path=pre["cache_path"],
                                   deadline=deadline)
    except Exception as exc:  # noqa: BLE001 - network/archive failure
        cause = prefetch_error or exc
        raise RunContractError(
            "gaia_quality_unavailable",
            f"the Gaia quality flags of {len(ids)} calibrator candidates "
            f"could not be obtained ({type(cause).__name__}: {cause}); the run "
            "stops before the fit instead of calibrating the ZP on the "
            "unfiltered set. Re-run when the Gaia archive answers, pre-fetch "
            "with `tract7dt augment-gaia <csv>`, or set "
            "zp.gaia_quality_filter: false",
            n_candidates=len(ids)) from exc
    missing = set(ids) - set(flags["source_id"].astype(str))
    if missing:
        raise RunContractError(
            "gaia_quality_unavailable",
            f"the Gaia quality cache lacks {len(missing)} of {len(ids)} "
            "calibrator candidates after the fetch",
            n_candidates=len(ids), n_missing=len(missing))
    waited = time.monotonic() - t0
    run_status.note("gaia_quality_flags", {
        "n_candidates": len(ids),
        "wait_before_fit_s": round(waited, 1),
        "prefetch_error": (None if prefetch_error is None
                           else str(prefetch_error))})
    logger.info("Gaia quality: flags of all %d calibrator candidates in hand "
                "(waited %.1f s before the fit)", len(ids), waited)


# ---------------------------------------------------------------------------
#  ZP computation
# ---------------------------------------------------------------------------

def _sigma_clip_mad(z: np.ndarray, clip_sigma: float, max_iters: int) -> np.ndarray:
    """Return boolean mask of points to keep after iterative MAD-based clipping."""
    keep = np.isfinite(z)
    for _ in range(int(max_iters)):
        zz = z[keep]
        if len(zz) < 3:
            break
        med = float(np.median(zz))
        mad = float(np.median(np.abs(zz - med)))
        sigma = 1.4826 * mad
        if sigma <= 0:
            break
        prev = keep.copy()
        keep = np.isfinite(z) & (np.abs(z - med) <= clip_sigma * sigma)
        if np.array_equal(keep, prev):
            break
    return keep


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median of values."""
    order = np.argsort(values)
    sv = values[order]
    sw = weights[order]
    cumw = np.cumsum(sw)
    half = cumw[-1] / 2.0
    idx = np.searchsorted(cumw, half)
    return float(sv[min(idx, len(sv) - 1)])


def _fit_spatial_zp(
    x: np.ndarray, y: np.ndarray, zp_i: np.ndarray, weights: np.ndarray,
    zp_med: float, *, model: str, poly_order: int, min_per_cell: int,
    grid: np.ndarray | None, x_extent: tuple[float, float],
    y_extent: tuple[float, float], clip_sigma: float = 3.0,
):
    """Fit a spatially-varying ZP as a correction delta(x,y) = zp_i - zp_med.

    A single scalar ZP per band leaves a position-dependent residual (a
    flat/illumination/coadd-level achromatic ZP pattern; measured ~0.015 mag
    RMS on the T08285 bulge field). This recovers it via an ubercal-style
    spatial model fit to the kept Gaia stars:
      - 'poly'     : robust 2D polynomial in normalized image coords.
      - 'per_grid' : per-grid (grid_id_fit) median offset; cells with
                     < min_per_cell stars fall back to the scalar ZP.
    Returns ``(evaluator, residual, info)`` where ``evaluator(xq, yq, gq)``
    returns the spatial ZP at query positions and ``residual`` is
    ``zp_i - evaluator(...)`` on the input sample (for the reported zp_err).
    Falls back to the scalar ``zp_med`` wherever position/grid is missing.
    """
    d = zp_i - zp_med
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
    m = str(model).lower()

    if m == "per_grid" and grid is not None:
        gg = np.asarray(grid, dtype=float)
        ok = finite & np.isfinite(gg) & (gg >= 0)
        offsets: dict[int, float] = {}
        for gid in np.unique(gg[ok]).astype(int):
            sel = ok & (gg == gid)
            if int(sel.sum()) >= int(min_per_cell):
                offsets[int(gid)] = float(np.median(d[sel]))
        max_g = max(offsets.keys()) if offsets else -1
        off_arr = np.zeros(max_g + 2, dtype=float)
        for gid, v in offsets.items():
            off_arr[gid] = v

        def evaluator(xq, yq, gq):
            gq = np.asarray(gq, dtype=float)
            out = np.full(gq.shape, zp_med, dtype=float)
            valid = np.isfinite(gq) & (gq >= 0) & (gq <= max_g)
            gi = gq[valid].astype(int)
            out[valid] = zp_med + off_arr[gi]
            return out

        gpred = np.array([off_arr[int(g)] if (np.isfinite(g) and 0 <= g <= max_g) else 0.0
                          for g in gg])
        return evaluator, (zp_i - (zp_med + gpred)), {
            "kind": "per_grid", "n_cells": len(offsets),
            "offsets": {str(g): float(v) for g, v in sorted(offsets.items())}}

    # --- 'poly' (default spatial model) ---
    # Guard: an order-N 2D poly has (N+1)(N+2)/2 terms; with too few stars the
    # fit is unconstrained at the field edges. Raising here lands in the
    # caller's fallback (warning + scalar ZP, summary records "scalar").
    n_terms_req = (int(poly_order) + 1) * (int(poly_order) + 2) // 2
    if int(finite.sum()) < 10 * n_terms_req:
        raise ValueError(
            f"too few calibrators for spatial poly fit "
            f"({int(finite.sum())} < {10 * n_terms_req} = 10x n_terms)")
    xc, xs = x_extent
    yc, ys = y_extent
    xs = xs if xs > 0 else 1.0
    ys = ys if ys > 0 else 1.0

    def design(xx, yy):
        X = (np.asarray(xx, dtype=float) - xc) / xs
        Y = (np.asarray(yy, dtype=float) - yc) / ys
        cols = [np.ones_like(X)]
        for o in range(1, int(poly_order) + 1):
            for i in range(o + 1):
                cols.append((X ** (o - i)) * (Y ** i))
        return np.column_stack(cols)

    A = design(x[finite], y[finite])
    dz = d[finite]
    w0 = weights[finite]
    w0 = np.where(np.isfinite(w0) & (w0 > 0), w0, 1.0)
    keep = np.ones(len(dz), dtype=bool)
    coef = np.zeros(A.shape[1])
    for _ in range(5):
        ww = np.sqrt(w0) * keep
        coef, *_ = np.linalg.lstsq(A * ww[:, None], dz * ww, rcond=None)
        r = dz - A @ coef
        med = float(np.median(r))
        s = 1.4826 * float(np.median(np.abs(r - med)))
        if s <= 0:
            break
        nk = np.abs(r - med) <= clip_sigma * s
        if np.array_equal(nk, keep):
            break
        keep = nk

    def evaluator(xq, yq, gq=None):
        # Honor the documented scalar fallback where positions are missing:
        # NaN coords would otherwise propagate into NaN magnitudes.
        val = zp_med + design(xq, yq) @ coef
        fin = (np.isfinite(np.asarray(xq, dtype=float))
               & np.isfinite(np.asarray(yq, dtype=float)))
        return np.where(fin, val, zp_med)

    resid = d.copy()
    resid[finite] = dz - A @ coef
    # coef/extents reproduce the surface offline (ZP/zp_spatial_model.json):
    # ZP(x, y) = zp_median + sum_k coef[k] * X^(o-i) * Y^i, X = (x - xc) / xs,
    # Y = (y - yc) / ys, terms ordered by total order o = 0..N, then i = 0..o.
    return evaluator, resid, {"kind": "poly", "order": int(poly_order), "n_terms": int(A.shape[1]),
                              "coef": [float(c) for c in coef],
                              "x_extent": [float(xc), float(xs)], "y_extent": [float(yc), float(ys)]}


def _zp_support(model_delta, kept_x, kept_y, kept_d, qx, qy, *, k: int, tol: float) -> dict:
    """R08: where is a band's spatial ZP model supported by its calibrators?

    ``model_delta`` is the model's correction (ZP - zp_median) at the query
    positions ``(qx, qy)``; ``kept_*`` are the kept calibrators' positions and
    their own corrections ``zp_i - zp_median``. A query is unsupported when it
    lies outside the calibrators' convex hull (the polynomial extrapolates
    there: -42 to +274 mag on the 2026-09 21-tile request sample), or when the
    model differs from the median of its ``k`` nearest calibrators by more than
    ``tol`` (the polynomial overshoots in sparsely calibrated patches inside the
    hull: 0.11-1.5 mag on the same sample, where those neighbours agree with
    each other). Queries without a finite position are never unsupported.

    Returns ``{"outside_hull", "mismatch"}`` (boolean arrays aligned with the
    queries; ``mismatch`` only inside the hull) and ``"hull"`` (the hull's
    vertices, ``[[x, y], ...]``; empty when no hull could be built, in which
    case every finite query is outside).
    """
    from scipy.spatial import ConvexHull, Delaunay, QhullError, cKDTree

    q = np.column_stack([np.asarray(qx, dtype=float), np.asarray(qy, dtype=float)])
    delta = np.asarray(model_delta, dtype=float)
    pts = np.column_stack([np.asarray(kept_x, dtype=float), np.asarray(kept_y, dtype=float)])
    dk = np.asarray(kept_d, dtype=float)
    good = np.isfinite(pts).all(axis=1) & np.isfinite(dk)
    pts, dk = pts[good], dk[good]
    fin = np.isfinite(q).all(axis=1)
    outside = np.zeros(len(q), dtype=bool)
    mismatch = np.zeros(len(q), dtype=bool)
    try:
        tri = Delaunay(pts)
        hull = pts[ConvexHull(pts).vertices].tolist()
    except (QhullError, ValueError):
        outside[fin] = True
        return {"outside_hull": outside, "mismatch": mismatch, "hull": []}
    outside[fin] = tri.find_simplex(q[fin]) < 0
    ins = fin & ~outside
    if np.any(ins):
        kk = int(min(max(int(k), 1), len(pts)))
        _, nn = cKDTree(pts).query(q[ins], k=kk)
        local = np.median(dk[np.asarray(nn).reshape(int(ins.sum()), -1)], axis=1)
        mismatch[ins] = np.abs(delta[ins] - local) > float(tol)
    return {"outside_hull": outside, "mismatch": mismatch, "hull": hull}


def _count_usable_bands(cat: pd.DataFrame, flag_cols: dict) -> np.ndarray:
    """``n_bands_usable`` as the band contract defines it (R06): bands with
    ``FLAGS_{band}_fit == 0`` and a finite flux."""
    n = np.zeros(len(cat), dtype=np.int16)
    for band, fc in flag_cols.items():
        fl_col = f"FLUX_{band}_fit"
        if fl_col not in cat.columns:
            continue
        fl = pd.to_numeric(cat[fl_col], errors="coerce").to_numpy(dtype=float)
        fg = pd.to_numeric(cat[fc], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        n += ((fg == 0) & np.isfinite(fl)).astype(np.int16)
    return n


def _write_zp_models(path: Path, models: dict, *, k: int, tol: float) -> None:
    """R08: save each band's zero point (scalar, or the spatial surface and its
    support), so a magnitude can be traced and re-evaluated offline."""
    import json

    from .provenance import code_identity
    doc = {
        "description": (
            "Per-band zero point. kind scalar: ZP = zp_median. Otherwise "
            "ZP(x, y) = zp_median + model (poly: "
            "coef over X = (x - x_extent[0]) / x_extent[1], Y likewise, terms by "
            "total order o = 0..order then i = 0..o as X^(o-i) Y^i; per_grid: "
            "offsets by grid id). Rows outside 'hull' (the kept calibrators' "
            "convex hull, pixel coordinates of x_pix_white_fit/y_pix_white_fit), "
            "or where the model differs from the median of the k_nearest kept "
            "calibrators by more than tolerance_mag, use zp_median and carry "
            "FLAGS bit 4."),
        "k_nearest": int(k),
        "tolerance_mag": float(tol),
        "code": code_identity(),
        "bands": models,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    logger.info("Wrote ZP models: %s", path)


def _gaia_cstar(excess: np.ndarray, bp_rp: np.ndarray) -> np.ndarray:
    """Corrected BP/RP flux-excess factor C* (Riello+2021, Sec 9.4): subtract
    the colour-dependent locus from ``phot_bp_rp_excess_factor``. |C*| near 0
    => consistent BP/RP; large |C*| flags blended/contaminated BP/RP (hence
    unreliable XP synphot), most common in crowded / low-b fields."""
    br = np.asarray(bp_rp, dtype=float)
    f = np.where(
        br < 0.5, 1.154360 + 0.033772 * br + 0.032277 * br ** 2,
        np.where(br < 4.0,
                 1.162004 + 0.011464 * br + 0.049255 * br ** 2 - 0.005879 * br ** 3,
                 1.057572 + 0.140537 * br))
    return np.asarray(excess, dtype=float) - f


def _gaia_cstar_sigma(g_mag: np.ndarray) -> np.ndarray:
    """Magnitude-dependent scatter of C* (Riello+2021), for sigma-based cuts."""
    g = np.asarray(g_mag, dtype=float)
    return 0.0059898 + 8.817481e-12 * np.power(g, 7.618399)


def _save_zp_band_plot(
    *,
    df: pd.DataFrame,
    band: str,
    zp_med: float,
    zp_mad: float,
    zp_std_sig: float | None = None,
    zp_err_snr_used: float | None = None,
    outpath: Path,
    dpi: int = 150,
    unreliable_rows: list[dict] | None = None,
    mag_lo: float = 10.0,
    mag_hi: float = 20.0,
    zp_offset: float | None = 0.25,
) -> None:
    """Save per-band ZP diagnostic plot (kept + clipped + unreliable).

    When *zp_offset* is a float the y-axis is fixed to ``[zp_med - offset,
    zp_med + offset]``.  When ``None`` the y-axis adapts to the data points
    (error bars are excluded from the range calculation).
    """
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))

    # --- Grey out-of-catalog-range vertical bands ---
    all_mags = df["gaiaxp_mag"].tolist()
    if unreliable_rows:
        all_mags.extend(r["gaiaxp_mag"] for r in unreliable_rows)
    mag_data_min = min(all_mags)
    mag_data_max = max(all_mags)
    if mag_data_min > mag_lo:
        plt.axvspan(mag_lo, mag_data_min, color="0.92", alpha=0.5, zorder=0)
    if mag_data_max < mag_hi:
        plt.axvspan(mag_data_max, mag_hi, color="0.92", alpha=0.5, zorder=0)

    # --- ZP error band (prefer std-based, fall back to MAD) ---
    _use_std = zp_std_sig is not None and np.isfinite(zp_std_sig)
    zp_disp_err = zp_std_sig if _use_std else zp_mad
    _err_label = "std" if _use_std else "MAD"
    if np.isfinite(zp_disp_err) and zp_disp_err > 0:
        shade = plt.axhspan(zp_med - zp_disp_err, zp_med + zp_disp_err,
                             color="0.8", alpha=0.35)
    else:
        shade = None

    # --- Unreliable sources (grey diamonds with error bars) ---
    if unreliable_rows:
        df_ur = pd.DataFrame(unreliable_rows)
        plt.errorbar(
            df_ur["gaiaxp_mag"].to_numpy(), df_ur["zp"].to_numpy(),
            yerr=df_ur["zp_err"].to_numpy(),
            fmt="d", ms=4, color="0.6", ecolor="0.6",
            elinewidth=0.7, capsize=0, alpha=0.6,
            label=f"unreliable (N={len(df_ur)})",
        )

    # --- Sigma-clipped sources (red X with error bars) ---
    d1 = df.loc[df["is_clipped"]]
    if len(d1):
        plt.errorbar(
            d1["gaiaxp_mag"].to_numpy(), d1["zp"].to_numpy(),
            yerr=d1["zp_err"].to_numpy(),
            fmt="x", ms=5, mew=1.2, color="tab:red", ecolor="tab:red",
            elinewidth=0.7, capsize=0, alpha=0.7,
            label=f"clipped (N={len(d1)})",
        )

    # --- Kept sources: split into high-SNR (black) and low-SNR (grey) ---
    d0 = df.loc[df["is_kept"]]
    if len(d0):
        if zp_err_snr_used is not None and np.isfinite(zp_err_snr_used):
            flux_k = d0["flux"].to_numpy()
            ferr_k = d0["fluxerr"].to_numpy()
            snr_k = np.where(
                np.isfinite(ferr_k) & (ferr_k > 0), flux_k / ferr_k, np.nan,
            )
            hi = np.isfinite(snr_k) & (snr_k >= zp_err_snr_used)
            d0_lo = d0.loc[~hi]
            d0_hi = d0.loc[hi]
        else:
            d0_lo = d0.iloc[:0]
            d0_hi = d0

        _has_snr_split = zp_err_snr_used is not None and np.isfinite(zp_err_snr_used)
        if len(d0_lo):
            plt.errorbar(
                d0_lo["gaiaxp_mag"].to_numpy(), d0_lo["zp"].to_numpy(),
                yerr=d0_lo["zp_err"].to_numpy(),
                fmt="o", ms=4, mfc="none", mec="0.55", ecolor="0.65",
                elinewidth=0.7, capsize=0, alpha=0.6,
                label=f"kept SNR<{zp_err_snr_used:.0f} (N={len(d0_lo)})",
            )
        if len(d0_hi):
            hi_label = (f"kept SNR>={zp_err_snr_used:.0f} (N={len(d0_hi)})"
                        if _has_snr_split else f"kept (N={len(d0_hi)})")
            plt.errorbar(
                d0_hi["gaiaxp_mag"].to_numpy(), d0_hi["zp"].to_numpy(),
                yerr=d0_hi["zp_err"].to_numpy(),
                fmt="o", ms=4, mfc="none", mec="k", ecolor="0.4",
                elinewidth=0.7, capsize=0, alpha=0.9,
                label=hi_label,
            )

    med_line = plt.axhline(zp_med, color="tab:red", lw=1.5,
                            label=f"median={zp_med:.3f} $\\pm$ {zp_disp_err:.3f}")
    if shade is not None:
        shade.set_label(f"median $\\pm$ {_err_label}")

    # --- Axis limits ---
    plt.xlim(mag_lo, mag_hi)
    if zp_offset is not None:
        plt.ylim(zp_med - zp_offset, zp_med + zp_offset)
    else:
        all_zps = df["zp"].tolist()
        if unreliable_rows:
            all_zps.extend(r["zp"] for r in unreliable_rows)
        zp_lo, zp_hi = min(all_zps), max(all_zps)
        pad = 0.05 * (zp_hi - zp_lo) if zp_hi > zp_lo else 0.1
        plt.ylim(zp_lo - pad, zp_hi + pad)

    plt.xlabel("GaiaXP synphot AB mag")
    plt.ylabel("ZP (mag)")
    plt.title(band)

    handles, labels = plt.gca().get_legend_handles_labels()
    desired_order = ["unreliable", "clipped", "kept SNR<", "kept SNR>", "kept", "median", "MAD"]
    ordered_hl = []
    for prefix in desired_order:
        for h, l in zip(handles, labels):
            if l.startswith(prefix) and not any(l == ol for _, ol in ordered_hl):
                ordered_hl.append((h, l))
    for h, l in zip(handles, labels):
        if not any(l == ol for _, ol in ordered_hl):
            ordered_hl.append((h, l))
    if ordered_hl:
        plt.legend([h for h, _ in ordered_hl], [l for _, l in ordered_hl],
                   loc="best", fontsize=8, framealpha=0.7)
    else:
        plt.legend(loc="best", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close()


def _save_zp_band_plots_worker(task: dict[str, Any]) -> tuple[str, str | None]:
    """Process-pool worker for diagnostic ZP plots."""
    try:
        band = str(task["band"])
        plot_kw = dict(task["plot_kw"])
        out_constrained = Path(task["out_constrained"])
        out_fullrange = Path(task["out_fullrange"])
        zp_offset = task["zp_offset"]
        _save_zp_band_plot(
            **plot_kw,
            outpath=out_constrained,
            zp_offset=zp_offset,
        )
        _save_zp_band_plot(
            **plot_kw,
            outpath=out_fullrange,
            zp_offset=None,
        )
        return band, None
    except Exception as exc:
        import traceback
        return str(task.get("band", "<unknown>")), (
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )


def compute_zp(
    *,
    cfg: dict,
    merged_catalog_path: Path,
    catalog_df: "pd.DataFrame | None" = None,
    gaia_synphot_df: "pd.DataFrame | None" = None,
) -> Path:
    """Compute per-band ZP from Gaia-matched stars and apply AB magnitudes.

    Overwrites the merged catalog with additional MAG/MAGERR columns.
    Returns the path to the updated catalog.

    ``catalog_df``: in gpu mode the grid stage hands the result catalog in
    memory and skips its own ``output_catalog.csv`` write (avoiding a redundant
    ~443 MB write + re-read); we then write the single final copy. When None
    (the CLI ``compute-zp`` path and per-patch mode) the merged catalog is read
    from ``merged_catalog_path`` as before -- the final result is identical.
    """
    zp_cfg = cfg["zp"]
    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    clip_sigma = float(zp_cfg.get("clip_sigma", 3.0))
    clip_max_iters = int(zp_cfg.get("clip_max_iters", 10))
    plot_mag_lo = float(zp_cfg.get("plot_mag_lo", 10.0))
    plot_mag_hi = float(zp_cfg.get("plot_mag_hi", 20.0))
    plot_zp_offset = float(zp_cfg.get("plot_zp_offset", 0.25))
    zp_err_method = str(zp_cfg.get("zp_err_method", "all_mad"))
    zp_err_snr_min = float(zp_cfg.get("zp_err_snr_min", 100.0))
    # Spatially-varying ZP (ubercal-style). "scalar" (default) = exact-legacy
    # single ZP per band. "poly"/"per_grid" recover the position-dependent
    # achromatic ZP structure (flat/illumination/coadd) that a scalar leaves
    # in the residual -- most relevant in crowded/low-galactic-latitude fields.
    spatial_zp_model = str(zp_cfg.get("spatial_zp_model", "scalar")).lower()
    spatial_zp_poly_order = int(zp_cfg.get("spatial_zp_poly_order", 4))
    spatial_zp_min_per_cell = int(zp_cfg.get("spatial_zp_min_per_cell", 30))
    # Parallel (byte-identical) CSV writer width for the big catalog/per-object
    # writes. pandas to_csv is single-threaded float->str formatting; row-blocks
    # format identically and concatenate exactly. 8 keeps plots(<=20)+write(8)
    # under the 32-core budget.
    csv_write_workers = int(zp_cfg.get("catalog_write_workers", 8))

    gaia_csv_path = inputs.get("gaiaxp_synphot_csv")
    passthrough = catalog_df is not None

    def _write_passthrough(df):
        # When the grid stage handed us the catalog in memory (and skipped its
        # own write), guarantee output_catalog.csv exists on EVERY early-exit
        # path -- otherwise a ZP bail-out would leave no catalog on disk. No-op
        # when reading from disk (the file is already present, as before).
        if passthrough:
            write_csv_parallel(df, merged_catalog_path, workers=csv_write_workers)
        return merged_catalog_path

    def _fail_uncalibrated(df, why):
        # R12: the ZP stage was asked to calibrate and cannot. The fit is
        # kept (the catalog is written with fluxes and no magnitudes, and
        # `tract7dt compute-zp` can finish it once the cause is fixed), but the
        # run fails instead of passing the catalog off as complete.
        logger.error("ZP not computed: %s", why)
        _write_passthrough(df)
        raise RunContractError(
            "zp_not_computed",
            f"{why}; {Path(merged_catalog_path).name} was written without "
            "magnitudes (run `tract7dt compute-zp` after fixing the cause)")

    # Load the catalog: in-memory handoff (gpu mode) or read from disk.
    cat = catalog_df if passthrough else pd.read_csv(merged_catalog_path, dtype={"gaia_source_id": str})

    if gaia_csv_path is None or not Path(gaia_csv_path).exists():
        _fail_uncalibrated(cat, f"GaiaXP synphot CSV not found ({gaia_csv_path})")
    # Reuse the synphot DataFrame already parsed by augment_catalog_with_gaia in
    # this process (same file, same dtype -> identical frame), avoiding a 2nd
    # ~560k-row read. Falls back to a fresh read for the standalone compute-zp CLI.
    if gaia_synphot_df is not None:
        gaia = gaia_synphot_df
        logger.info("Reusing in-memory GaiaXP synphot frame (%d rows; skipped re-read)", len(gaia))
    else:
        gaia = pd.read_csv(gaia_csv_path, dtype={"source_id": str})

    if "gaia_source_id" not in cat.columns:
        _fail_uncalibrated(cat, "no gaia_source_id column in the merged catalog")

    cat["gaia_source_id"] = cat["gaia_source_id"].fillna("").astype(str).str.strip()
    has_gaia = (cat["gaia_source_id"] != "") & (cat["gaia_source_id"] != "nan")

    if "fit_is_reliable" not in cat.columns:
        _fail_uncalibrated(cat, "no fit_is_reliable column in the merged catalog")

    reliable = cat["fit_is_reliable"].fillna(False).astype(bool)
    # A calibrator must be clean in EVERY band (R11): a source with a saturated
    # pixel within source_saturation_cut.radius_pix in any band (FLAGS bit 8)
    # does not calibrate any band. Since saturation became a per-band
    # exclusion, these bright stars are fitted in their other bands, and their
    # tiny statistical errors let them carry 84-99% of the inverse-variance
    # weight of the ZP median (T08676, ~150-200 stars per band), moving every
    # band's ZP by up to 26 mmag; dev7 never used them (it dropped the source).
    sat_any = saturated_in_any_band(cat)
    n_sat_cal = int((has_gaia & reliable & sat_any).sum())
    n_gaia_total = int(has_gaia.sum())
    n_unreliable = int((has_gaia & ~reliable).sum())
    gaia_sources = cat[has_gaia & reliable & ~sat_any].copy()
    gaia_unreliable = cat[has_gaia & ~reliable].copy() if n_unreliable > 0 else None
    logger.info("Gaia-matched sources for ZP: %d (excluded %d unreliable and %d "
                "near saturation in some band, out of %d)",
                len(gaia_sources), n_unreliable, n_sat_cal, n_gaia_total)

    if len(gaia_sources) == 0:
        _fail_uncalibrated(cat, "no reliable Gaia-matched calibrator")

    gaia_lookup = gaia.drop_duplicates(subset="source_id").set_index("source_id")

    # Gaia-XP reference quality filter (RUWE + corrected C*), DEFAULT ON. Excludes
    # blended/contaminated XP calibrators from the ZP-DERIVATION sample only -- it
    # does NOT change injection (inject_gaia_sources) or which sources are fit/
    # output. The flags come from the GaiaXP CSV, else from the run's cache
    # (filled before the fit, require_gaia_quality_flags), else from the Gaia
    # archive within one time budget (the `tract7dt compute-zp` re-run). If
    # they cannot be obtained the ZP is not computed (R29) -- it used to be
    # calibrated on the UNFILTERED set while the run reported "ok".
    # If the cut would leave fewer than gaia_min_calibrators, it falls back to the
    # UNFILTERED set rather than calibrating on a thin sample.
    qf_active = False
    qf_n_excluded = 0
    qf_max_ruwe = float(zp_cfg.get("gaia_max_ruwe", 1.4))
    qf_cstar_nsig = float(zp_cfg.get("gaia_cstar_nsigma", 3.0))
    if bool(zp_cfg.get("gaia_quality_filter", True)):
        need = {"ruwe", "phot_bp_rp_excess_factor", "bp_rp", "phot_g_mean_mag"}
        if not {"bp_rp", "phot_g_mean_mag"} <= set(gaia_lookup.columns):
            _fail_uncalibrated(
                cat, "zp.gaia_quality_filter is on but the GaiaXP CSV lacks "
                f"{sorted({'bp_rp', 'phot_g_mean_mag'} - set(gaia_lookup.columns))} "
                "(needed for the corrected BP/RP excess); set "
                "zp.gaia_quality_filter: false to calibrate without the filter")
        flag_cols = {"ruwe", "phot_bp_rp_excess_factor"}
        if not flag_cols <= set(gaia_lookup.columns):
            from .gaia_quality import (QUALITY_BUDGET_S, QUALITY_COLS,
                                       load_quality_flags)
            cache = Path(outputs["work_dir"]) / "ZP" / "gaia_quality_cache.csv"
            cal_ids = gaia_sources["gaia_source_id"].astype(str).unique()
            try:
                flags = load_quality_flags(
                    cal_ids, cache_path=str(cache),
                    deadline=time.monotonic() + QUALITY_BUDGET_S)
            except Exception as exc:  # noqa: BLE001 - network/archive failure
                _fail_uncalibrated(
                    cat, f"the Gaia quality flags of {len(cal_ids)} calibrators "
                    f"could not be obtained ({type(exc).__name__}: {exc}); "
                    "pre-fetch with `tract7dt augment-gaia <csv>` or set "
                    "zp.gaia_quality_filter: false")
            gaia_lookup = gaia_lookup.drop(
                columns=[c for c in QUALITY_COLS if c in gaia_lookup.columns],
                errors="ignore").join(flags.set_index("source_id")[QUALITY_COLS])
        ruwe = pd.to_numeric(gaia_lookup["ruwe"], errors="coerce").to_numpy(dtype=float)
        cstar = _gaia_cstar(
            pd.to_numeric(gaia_lookup["phot_bp_rp_excess_factor"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(gaia_lookup["bp_rp"], errors="coerce").to_numpy(dtype=float))
        sig = _gaia_cstar_sigma(pd.to_numeric(gaia_lookup["phot_g_mean_mag"], errors="coerce").to_numpy(dtype=float))
        bad = ((np.isfinite(ruwe) & (ruwe > qf_max_ruwe))
               | (np.isfinite(cstar) & np.isfinite(sig) & (np.abs(cstar) > qf_cstar_nsig * sig)))
        bad_ids = set(gaia_lookup.index[bad].astype(str))
        n_before = len(gaia_sources)
        keep_mask = ~gaia_sources["gaia_source_id"].isin(bad_ids)
        n_kept = int(keep_mask.sum())
        min_cal = int(zp_cfg.get("gaia_min_calibrators", 10))
        if n_kept < min_cal:
            logger.warning(
                "Gaia quality filter would leave only %d calibrators "
                "(< gaia_min_calibrators=%d); falling back to the UNFILTERED set "
                "of %d for ZP.", n_kept, min_cal, n_before)
        else:
            gaia_sources = gaia_sources[keep_mask].copy()
            qf_active = True
            qf_n_excluded = int((~keep_mask).sum())
            logger.info("Gaia quality filter (RUWE>%.2f or |C*|>%.1f sigma): "
                        "excluded %d/%d ZP calibrators", qf_max_ruwe,
                        qf_cstar_nsig, qf_n_excluded, n_before)

    # A photometric band is a FLUX_{band}_fit column that has its own
    # FLUXERR_{band}_fit. Disclosure columns sharing the prefix
    # (FLUX_WIN_{band}_fit, the data-constrained window flux; FLUX_MARG_FRAC_
    # {band}_fit) have no error column, so they never become pseudo-bands
    # (which emitted all-NaN MAG_WIN_*, and until R14 MAG_MARG_FRAC_* /
    # MAGERR_MARG_FRAC_* columns: 44 on T08676, 40 on T08285). A window
    # magnitude, if needed, is ZP_band - 2.5*log10(FLUX_WIN_band).
    flux_pat = re.compile(r"^FLUX_(.+)_fit$")
    bandnames = sorted(m.group(1) for c in cat.columns for m in [flux_pat.match(c)]
                       if m and f"FLUXERR_{m.group(1)}_fit" in cat.columns)
    logger.info("Bands for ZP: %s", ", ".join(bandnames))

    zp_dir = Path(outputs["work_dir"]) / "ZP"
    zp_dir.mkdir(parents=True, exist_ok=True)

    # Spatial-ZP setup (positions + grid + normalization extent).
    band_zp_models: dict[str, Any] = {}
    xcol = ycol = gcol = None
    x_extent = y_extent = (0.0, 1.0)
    do_spatial = spatial_zp_model != "scalar"
    if do_spatial:
        xcol = "x_pix_white_fit" if "x_pix_white_fit" in cat.columns else (
            "x_pix_white" if "x_pix_white" in cat.columns else None)
        ycol = "y_pix_white_fit" if "y_pix_white_fit" in cat.columns else (
            "y_pix_white" if "y_pix_white" in cat.columns else None)
        gcol = "grid_id_fit" if "grid_id_fit" in cat.columns else None
        if xcol is None or ycol is None or (spatial_zp_model == "per_grid" and gcol is None):
            logger.warning("spatial_zp_model=%s requested but required columns missing "
                           "(x=%s y=%s grid=%s); falling back to scalar ZP",
                           spatial_zp_model, xcol, ycol, gcol)
            do_spatial = False
    if do_spatial:
        # Catalog positions/grid for spatial-ZP evaluation (once for all bands).
        cat_x = _xv = pd.to_numeric(cat[xcol], errors="coerce").to_numpy(dtype=float)
        cat_y = _yv = pd.to_numeric(cat[ycol], errors="coerce").to_numpy(dtype=float)
        cat_grid = (pd.to_numeric(cat[gcol], errors="coerce").to_numpy(dtype=float)
                    if gcol else np.full(len(cat), np.nan))
        _xf = _xv[np.isfinite(_xv)]; _yf = _yv[np.isfinite(_yv)]
        if len(_xf) and len(_yf):
            x_extent = (0.5 * (float(_xf.min()) + float(_xf.max())),
                        max(0.5 * (float(_xf.max()) - float(_xf.min())), 1.0))
            y_extent = (0.5 * (float(_yf.min()) + float(_yf.max())),
                        max(0.5 * (float(_yf.max()) - float(_yf.min())), 1.0))
        logger.info("Spatial ZP: model=%s poly_order=%d min_per_cell=%d",
                    spatial_zp_model, spatial_zp_poly_order, spatial_zp_min_per_cell)

    summary_rows: list[dict] = []
    all_zp_rows: list[pd.DataFrame] = []
    plot_tasks: list[dict[str, Any]] = []
    # R08: per band, the ZP at every catalog row (model where supported,
    # scalar zp_median elsewhere), the rows that fell back, the ZP error for
    # those rows, and the saved model.
    band_zp_at: dict[str, np.ndarray] = {}
    band_unsupported: dict[str, np.ndarray] = {}
    band_zp_err_fallback: dict[str, float] = {}
    spatial_models: dict[str, dict] = {}

    zp_missing: dict[str, str] = {}   # R12: bands left without a ZP, and why
    for band in bandnames:
        flux_col = f"FLUX_{band}_fit"
        ferr_col = f"FLUXERR_{band}_fit"
        flags_col = f"FLAGS_{band}_fit"
        gaia_mag_col = f"mag_{band}"

        if flux_col not in gaia_sources.columns:
            zp_missing[band] = "no fitted flux column"
            continue
        if gaia_mag_col not in gaia_lookup.columns:
            logger.info("[%s] no mag_%s in GaiaXP CSV; skip", band, band)
            zp_missing[band] = f"no mag_{band} in the GaiaXP CSV"
            continue

        def _build_zp_rows(src_df):
            """Vectorized build of per-source ZP rows (replaces an iterrows() +
            per-row ``gaia_lookup.loc[gid]`` label lookup that cost ~8.7 s/band).
            Returns a DataFrame in ``src_df`` row order with invalid rows dropped;
            bit-identical to the legacy row-by-row construction."""
            cols_order = (["gaia_source_id", "gaiaxp_mag", "flux", "fluxerr", "zp", "zp_err",
                           "band_flags"]
                          + (["x", "y", "grid"] if do_spatial else []))
            n = len(src_df)
            if n == 0:
                return pd.DataFrame(columns=cols_order)
            gid = src_df["gaia_source_id"].astype(str).str.strip()
            gid_arr = gid.to_numpy()
            # Per-row gaia_lookup.loc[gid] -> single vectorized hash join. gids not
            # present in gaia_lookup map to NaN and are dropped by the finite mask,
            # exactly as the old `gid not in gaia_lookup.index: continue` did.
            gaia_mag = pd.to_numeric(gid.map(gaia_lookup[gaia_mag_col]), errors="coerce").to_numpy(dtype=float)
            flux = (pd.to_numeric(src_df[flux_col], errors="coerce").to_numpy(dtype=float)
                    if flux_col in src_df.columns else np.full(n, np.nan))
            if ferr_col in src_df.columns:
                fluxerr = pd.to_numeric(src_df[ferr_col], errors="coerce").to_numpy(dtype=float)
                fluxerr = np.where(np.isfinite(fluxerr), fluxerr, np.nan)  # match old: non-finite -> NaN
            else:
                fluxerr = np.full(n, np.nan)
            mask = ((gid_arr != "") & (gid_arr != "nan")
                    & np.isfinite(gaia_mag) & np.isfinite(flux) & (flux > 0))
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return pd.DataFrame(columns=cols_order)
            flux_m = flux[idx]
            fluxerr_m = fluxerr[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                zp_err_m = np.where(np.isfinite(fluxerr_m) & (fluxerr_m > 0),
                                    K_MAG * (fluxerr_m / flux_m), np.nan)
            # The band's reason flags (R06) without this stage's own bit 4.
            bflags = (pd.to_numeric(src_df[flags_col], errors="coerce").fillna(0)
                      .to_numpy(dtype=np.int64)[idx] & ~FLAG_ZP_EXTRAP
                      if flags_col in src_df.columns else np.zeros(idx.size, dtype=np.int64))
            cols = {
                "gaia_source_id": gid_arr[idx],
                "gaiaxp_mag": gaia_mag[idx],
                "flux": flux_m,
                "fluxerr": fluxerr_m,
                "zp": gaia_mag[idx] + 2.5 * np.log10(flux_m),
                "zp_err": zp_err_m,
                "band_flags": bflags,
            }
            if do_spatial:
                cols["x"] = (pd.to_numeric(src_df[xcol], errors="coerce").to_numpy(dtype=float)[idx]
                             if xcol in src_df.columns else np.full(idx.size, np.nan))
                cols["y"] = (pd.to_numeric(src_df[ycol], errors="coerce").to_numpy(dtype=float)[idx]
                             if ycol in src_df.columns else np.full(idx.size, np.nan))
                cols["grid"] = (pd.to_numeric(src_df[gcol], errors="coerce").to_numpy(dtype=float)[idx]
                                if (gcol and gcol in src_df.columns) else np.full(idx.size, np.nan))
            return pd.DataFrame(cols)

        df = _build_zp_rows(gaia_sources)
        df_unrel = _build_zp_rows(gaia_unreliable) if gaia_unreliable is not None else None
        # Plot worker consumes unreliable rows as a list of dicts (it iterates and
        # rebuilds a DataFrame), so keep that interface for the (small) unreliable set.
        rows_unreliable = (df_unrel.to_dict("records")
                           if (df_unrel is not None and len(df_unrel)) else [])
        n_unrel = len(rows_unreliable)

        if len(df) == 0:
            logger.info("[%s] no valid ZP rows", band)
            zp_missing[band] = "no valid calibrator row"
            continue

        # Optional S/N floor for ZP DERIVATION (zp.min_calibrator_snr,
        # default 0 = off, exact-legacy). Faint calibrators carry both
        # photon noise and a one-sided Gaia-XP crowding bias; with the
        # inverse-variance weighted median they barely move the ZP value
        # (measured <=0.0013 mag on T08285) but inflate the reported
        # scatter ~25%. Cut rows still appear in the per-object table and
        # plots as clipped. NOTE: this is calibrator BOOKKEEPING, not a
        # photometry improvement -- do not read a kept-MAD drop from this
        # cut as better fitting.
        min_snr = float(zp_cfg.get("min_calibrator_snr", 0.0))
        snr_ok = np.ones(len(df), dtype=bool)
        if min_snr > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                _snr = (pd.to_numeric(df["flux"], errors="coerce")
                        / pd.to_numeric(df["fluxerr"], errors="coerce")).to_numpy(dtype=float)
            snr_ok = np.isfinite(_snr) & (_snr >= min_snr)
            logger.info("[%s] ZP derivation S/N>=%g cut: %d / %d calibrators pass",
                        band, min_snr, int(snr_ok.sum()), len(df))

        # R08: a calibrator whose value in this band carries a reason flag
        # (R06; in practice the PSF-lobe zone, since no-data and saturated
        # cells have no flux) does not calibrate the band. Until dev8 only the
        # sigma clip kept them out: lobe-zone calibrators read 0.06-0.8 mag
        # faint (R07), and those inside the clip pulled the spatial ZP where
        # the lobes are, near the field edge. They stay in the per-object
        # table as not kept (band_flags column).
        flag_ok = df["band_flags"].to_numpy(dtype=np.int64) == 0
        n_flag_excl = int((snr_ok & ~flag_ok).sum())
        if n_flag_excl:
            logger.info("[%s] ZP derivation: %d calibrator(s) with a band reason "
                        "flag excluded", band, n_flag_excl)
        cand = snr_ok & flag_ok
        keep = cand.copy()
        if cand.any():
            keep[cand] = _sigma_clip_mad(
                df["zp"].to_numpy()[cand], clip_sigma, clip_max_iters)
        df["is_kept"] = keep
        df["is_clipped"] = ~keep

        df_keep = df.loc[df["is_kept"]].copy()
        if len(df_keep) == 0:
            logger.info("[%s] all points clipped; skip", band)
            zp_missing[band] = "every calibrator clipped or flagged"
            continue

        zp_vals = df_keep["zp"].to_numpy()
        zp_errs = df_keep["zp_err"].to_numpy()
        valid_err = np.isfinite(zp_errs) & (zp_errs > 0)

        if np.any(valid_err):
            weights = np.where(valid_err, 1.0 / (zp_errs ** 2), 0.0)
            if weights.sum() > 0:
                zp_med = _weighted_median(zp_vals[valid_err], weights[valid_err])
            else:
                zp_med = float(np.median(zp_vals))
        else:
            zp_med = float(np.median(zp_vals))

        zp_mad = float(np.median(np.abs(zp_vals - zp_med)))
        zp_err_mad_scalar = zp_mad
        zp_raw = zp_vals

        # Spatial ZP: fit delta(x,y) around the scalar zp_med and report the
        # scatter on the spatially-corrected residual. zp_vals is reassigned
        # to the residual so the downstream bright-std uses it too.
        if do_spatial:
            xk = df_keep["x"].to_numpy(dtype=float) if "x" in df_keep.columns else np.full(len(df_keep), np.nan)
            yk = df_keep["y"].to_numpy(dtype=float) if "y" in df_keep.columns else np.full(len(df_keep), np.nan)
            gk = df_keep["grid"].to_numpy(dtype=float) if "grid" in df_keep.columns else None
            ze = df_keep["zp_err"].to_numpy(dtype=float)
            wk = np.where(np.isfinite(ze) & (ze > 0), 1.0 / np.maximum(ze, 1e-6) ** 2, 1.0)
            try:
                zp_model_fn, resid_keep, sinfo = _fit_spatial_zp(
                    xk, yk, zp_vals, wk, zp_med,
                    model=spatial_zp_model, poly_order=spatial_zp_poly_order,
                    min_per_cell=spatial_zp_min_per_cell, grid=gk,
                    x_extent=x_extent, y_extent=y_extent, clip_sigma=clip_sigma)
                band_zp_models[band] = zp_model_fn
                rc = resid_keep[np.isfinite(resid_keep)]
                if len(rc):
                    zp_mad = float(np.median(np.abs(rc - np.median(rc))))
                    zp_vals = resid_keep
                logger.info("[%s] spatial ZP %s: zp_err_mad %.4f -> %.4f",
                            band, sinfo, zp_err_mad_scalar, zp_mad)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[%s] spatial ZP fit failed (%s); using scalar", band, exc)

        # R08: the spatial model only where its calibrators support it. Every
        # row with a measured error gets a ZP (MAG and MAG_UL5); outside the
        # kept calibrators' hull, or where the model and the median of its
        # nearest calibrators disagree by more than ZP_SUPPORT_TOL_MAG, the
        # row gets the scalar zp_med and FLAGS bit 4 (ZP_EXTRAP).
        n_zp_hull = n_zp_local = 0
        if do_spatial and band in band_zp_models:
            zp_at = band_zp_models[band](cat_x, cat_y, cat_grid)
            ferr_all = (pd.to_numeric(cat[ferr_col], errors="coerce").to_numpy(dtype=float)
                        if ferr_col in cat.columns else np.full(len(cat), np.nan))
            need = np.isfinite(ferr_all) & np.isfinite(cat_x) & np.isfinite(cat_y)
            sup = _zp_support(
                zp_at[need] - zp_med, xk, yk, zp_raw - zp_med, cat_x[need], cat_y[need],
                k=spatial_zp_min_per_cell, tol=ZP_SUPPORT_TOL_MAG)
            unsup = np.zeros(len(cat), dtype=bool)
            unsup[need] = sup["outside_hull"] | sup["mismatch"]
            band_zp_at[band] = np.where(unsup, zp_med, zp_at)
            band_unsupported[band] = unsup
            n_zp_hull = int(sup["outside_hull"].sum())
            n_zp_local = int(sup["mismatch"].sum())
            logger.info(
                "[%s] ZP support: of %d rows with a measured error, %d lie outside "
                "the calibrator hull and %d inside where the model differs from the "
                "median of its %d nearest calibrators by > %.2f mag -> scalar ZP + "
                "FLAGS bit %d", band, int(need.sum()), n_zp_hull, n_zp_local,
                spatial_zp_min_per_cell, ZP_SUPPORT_TOL_MAG, FLAG_ZP_EXTRAP)
            spatial_models[band] = {
                **sinfo, "zp_median": float(zp_med), "hull": sup["hull"],
                "n_rows_measured": int(need.sum()), "n_rows_outside_hull": n_zp_hull,
                "n_rows_local_mismatch": n_zp_local}

        flux_keep = df_keep["flux"].to_numpy()
        ferr_keep = df_keep["fluxerr"].to_numpy()
        snr_keep = np.where(
            np.isfinite(ferr_keep) & (ferr_keep > 0),
            flux_keep / ferr_keep,
            np.nan,
        )
        min_snr_sources = 10
        zp_err_std_val = np.nan
        zp_err_std_scalar = np.nan
        zp_err_n_src = 0
        zp_err_snr_used = np.nan
        for snr_thresh in (zp_err_snr_min, zp_err_snr_min / 2, zp_err_snr_min / 4):
            high_snr = np.isfinite(snr_keep) & (snr_keep >= snr_thresh)
            n_high = int(high_snr.sum())
            if n_high >= min_snr_sources:
                zp_err_std_val = float(np.std(zp_vals[high_snr]))
                # the same stars' RMS about the scalar ZP: the error of a row
                # that falls back to it (R08)
                zp_err_std_scalar = float(np.sqrt(np.mean((zp_raw[high_snr] - zp_med) ** 2)))
                zp_err_n_src = n_high
                zp_err_snr_used = snr_thresh
                if snr_thresh < zp_err_snr_min:
                    logger.warning(
                        "[%s] ZP error: using SNR>=%.1f (%d sources) after fallback",
                        band, snr_thresh, n_high,
                    )
                break
            logger.warning(
                "[%s] ZP error: only %d sources with SNR>=%.1f (need >=%d)",
                band, n_high, snr_thresh, min_snr_sources,
            )
        else:
            zp_err_std_val = zp_mad
            zp_err_std_scalar = zp_err_mad_scalar
            zp_err_n_src = int(len(df_keep))
            zp_err_snr_used = np.nan
            logger.warning(
                "[%s] ZP error: falling back to MAD-based scatter (%.4f)",
                band, zp_mad,
            )
        band_zp_err_fallback[band] = (zp_err_std_scalar if zp_err_method == "bright_std"
                                      else zp_err_mad_scalar)

        # QA: robust zp-vs-mag slope over the kept calibrators. A healthy
        # band is flat (|slope| <~ 0.01 mag/mag); a large slope means the
        # photometry (or the image itself) is magnitude-dependent broken --
        # e.g. T08285 m475 showed ~1.1 mag/mag while still passing every
        # other summary metric. Median-binned LSQ so single outliers and
        # the faint tail cannot fake or hide a slope.
        zp_mag_slope = float("nan")
        if len(df_keep) >= 50:
            _mags = df_keep["gaiaxp_mag"].to_numpy(dtype=float)
            _zps = df_keep["zp"].to_numpy(dtype=float)
            _bins = np.linspace(np.nanquantile(_mags, 0.02),
                                np.nanquantile(_mags, 0.98), 9)
            _centers, _medians = [], []
            for _lo, _hi in zip(_bins[:-1], _bins[1:]):
                _in = (_mags >= _lo) & (_mags < _hi)
                if _in.sum() >= 10:
                    _centers.append(0.5 * (_lo + _hi))
                    _medians.append(float(np.median(_zps[_in])))
            if len(_centers) >= 4:
                zp_mag_slope = float(np.polyfit(_centers, _medians, 1)[0])
                if abs(zp_mag_slope) > 0.02:
                    logger.warning(
                        "[%s] ZP QA: zp-vs-mag slope %.4f mag/mag exceeds 0.02 "
                        "-- magnitude-dependent photometry problem in this band "
                        "(check the band image / PSF model before trusting it)",
                        band, zp_mag_slope,
                    )

        summary_rows.append({
            "band": band,
            "n_raw": int(len(df)),
            "n_kept": int(len(df_keep)),
            "n_clipped": int((~keep).sum()),
            "n_unreliable": n_unrel,
            "clip_sigma": float(clip_sigma),
            "zp_median": zp_med,
            "zp_mag_slope": zp_mag_slope,
            "zp_err_mad": zp_mad,
            "zp_err_mad_scalar": zp_err_mad_scalar,
            "spatial_zp_model": spatial_zp_model if (do_spatial and band in band_zp_models) else "scalar",
            "n_zp_extrap_hull": n_zp_hull,
            "n_zp_extrap_local": n_zp_local,
            "zp_err_fallback": band_zp_err_fallback[band],
            "n_calibrators_flag_excluded": n_flag_excl,
            "gaia_quality_filter_active": bool(qf_active),
            "gaia_n_calibrators_excluded": int(qf_n_excluded),
            "gaia_max_ruwe": qf_max_ruwe,
            "gaia_cstar_nsigma": qf_cstar_nsig,
            "zp_err_std": zp_err_std_val,
            "n_zp_err_src": zp_err_n_src,
            "zp_err_snr_used": zp_err_snr_used,
        })

        df.insert(0, "band", band)
        all_zp_rows.append(df)

        plot_kw = dict(
            df=df, band=band, zp_med=zp_med, zp_mad=zp_mad,
            zp_std_sig=zp_err_std_val if zp_err_method == "bright_std" else None,
            zp_err_snr_used=zp_err_snr_used if zp_err_method == "bright_std" else None,
            dpi=int(cfg.get("plotting", {}).get("dpi", 150)),
            unreliable_rows=rows_unreliable if rows_unreliable else None,
            mag_lo=plot_mag_lo, mag_hi=plot_mag_hi,
        )
        plot_tasks.append({
            "band": band,
            "plot_kw": plot_kw,
            "out_constrained": str(zp_dir / f"zp_vs_mag__{band}.png"),
            "out_fullrange": str(zp_dir / f"zp_vs_mag__{band}_fullrange.png"),
            "zp_offset": plot_zp_offset,
        })
        snr_info = (f"N={zp_err_n_src}, SNR>={zp_err_snr_used:.0f}"
                    if np.isfinite(zp_err_snr_used) else "MAD fallback")
        active_err = zp_err_std_val if zp_err_method == "bright_std" else zp_mad
        logger.info("[%s] ZP=%.4f | err=%.4f [%s] (bright_std=%.4f (%s), all_mad=%.4f) | kept=%d unreliable=%d clipped=%d",
                     band, zp_med, active_err, zp_err_method,
                     zp_err_std_val, snr_info, zp_mad,
                     int(len(df_keep)), n_unrel, int((~keep).sum()))

    # Diagnostic plots are write-only and depend only on each band's plot_kw
    # (already built above). Submit them to the worker pool NOW and let them
    # render while the main thread does the (dominant) per-object + 443 MB
    # catalog CSV writes; drain just before returning. Identical plots/outputs,
    # only the dispatch timing changes.
    plot_ex = None
    plot_futs: list = []
    t_plot = time.perf_counter()
    if plot_tasks:
        # Plots are co-critical with the (now parallelized) catalog writes, so
        # give them more workers. Default cap raised from 8 to 20; combined with
        # the 8-wide parallel CSV writer this peaks at ~28 procs (< 32-core
        # budget). Identical PNG outputs -- only dispatch parallelism changes.
        plot_cap = int(zp_cfg.get("plot_workers", 0)) or min(len(plot_tasks), 20)
        plot_workers = min(len(plot_tasks), max(1, os.cpu_count() or 1), plot_cap)
        logger.info(
            "ZP diagnostic plots: rendering %d band plot pair(s) with %d worker "
            "process(es), overlapped with catalog writes", len(plot_tasks), plot_workers,
        )
        if plot_workers > 1:
            ctx = mp.get_context("spawn")
            plot_ex = ProcessPoolExecutor(max_workers=plot_workers, mp_context=ctx)
            plot_futs = [plot_ex.submit(_save_zp_band_plots_worker, task) for task in plot_tasks]

    if all_zp_rows:
        zp_long = pd.concat(all_zp_rows, ignore_index=True)
        write_csv_parallel(zp_long, zp_dir / "zp_all_bands__per_object.csv",
                           workers=csv_write_workers)
        logger.info("Wrote per-object ZP: %s", zp_dir / "zp_all_bands__per_object.csv")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(zp_dir / "zp_summary.csv", index=False)
        logger.info("Wrote ZP summary: %s", zp_dir / "zp_summary.csv")

        _err_key = "zp_err_std" if zp_err_method == "bright_std" else "zp_err_mad"
        zp_map = {row["band"]: (row["zp_median"], row[_err_key]) for row in summary_rows}

        # R08: this stage owns FLAGS bit 4 (ZP_EXTRAP). Clear it first, so a
        # `tract7dt compute-zp` re-run starts from the fit's band contract, set
        # it where the ZP fell back to the scalar, then recount usable bands.
        flag_cols = {b: f"FLAGS_{b}_fit" for b in bandnames if f"FLAGS_{b}_fit" in cat.columns}
        for fc in flag_cols.values():
            cat[fc] = (pd.to_numeric(cat[fc], errors="coerce").fillna(0)
                       .to_numpy(dtype=np.int64) & ~FLAG_ZP_EXTRAP).astype(np.int16)
        nbu_before = _count_usable_bands(cat, flag_cols)

        mag_updates: dict[str, np.ndarray] = {}
        for band in bandnames:
            mag_col = f"MAG_{band}_fit"
            magerr_col = f"MAGERR_{band}_fit"
            if band not in zp_map:
                # Same columns as a calibrated band, all NaN (a stable schema).
                nan_cols = [mag_col, magerr_col, f"MAG_UL5_{band}_fit"]
                if f"FLUXERR_TOT_{band}_fit" in cat.columns:
                    nan_cols.append(f"MAGERR_TOT_{band}_fit")
                if f"FLUXERR_MARG_TOT_{band}_fit" in cat.columns:
                    nan_cols.append(f"MAG_UL5_MARG_{band}_fit")
                for _src, _dst in ((f"FLUXERR_MARG_{band}_fit", f"MAGERR_MARG_{band}_fit"),
                                   (f"FLUXERR_MARG_TOT_{band}_fit",
                                    f"MAGERR_MARG_TOT_{band}_fit")):
                    if _src in cat.columns:
                        nan_cols.append(_dst)
                for c in nan_cols:
                    mag_updates[c] = np.full(len(cat), np.nan, dtype=float)
                continue

            zp, zp_err = zp_map[band]
            flux_col = f"FLUX_{band}_fit"
            ferr_col = f"FLUXERR_{band}_fit"

            flux = pd.to_numeric(cat[flux_col], errors="coerce").to_numpy(dtype=float) if flux_col in cat.columns else np.full(len(cat), np.nan)
            ferr = pd.to_numeric(cat[ferr_col], errors="coerce").to_numpy(dtype=float) if ferr_col in cat.columns else np.full(len(cat), np.nan)

            # The ZP of every row (spatial model where supported, scalar
            # elsewhere; R08) and the ZP error it carries: the model's, or the
            # scalar's scatter where the row fell back to the scalar.
            zp_at_all = band_zp_at.get(band)
            if zp_at_all is None:
                zp_at_all = np.full(len(cat), float(zp))
            zp_err_row = np.full(len(cat), float(zp_err))
            unsup = band_unsupported.get(band)
            if unsup is not None and np.any(unsup):
                zp_err_row[unsup] = float(band_zp_err_fallback[band])
                if band in flag_cols:
                    fc = flag_cols[band]
                    cat[fc] = (cat[fc].to_numpy(dtype=np.int64)
                               | np.where(unsup, FLAG_ZP_EXTRAP, 0)).astype(np.int16)

            # A magnitude only for a measured flux: finite and positive, with
            # a defined statistical error. A NaN FLUXERR means the band had
            # no information for the source (R05) -- no MAG.
            ok = (np.isfinite(flux) & (flux > 0)
                  & np.isfinite(ferr) & (ferr > 0))
            mag = np.full(len(cat), np.nan, dtype=float)
            magerr = np.full(len(cat), np.nan, dtype=float)
            mag[ok] = zp_at_all[ok] - 2.5 * np.log10(flux[ok])

            magerr[ok] = np.sqrt((K_MAG * (ferr[ok] / flux[ok])) ** 2 + zp_err_row[ok] ** 2)

            mag_updates[mag_col] = mag
            mag_updates[magerr_col] = magerr

            # Honest TOTAL magnitude error from the #27 confusion-corrected flux
            # error (FLUXERR_TOT = sqrt(FLUXERR^2 + FLUXERR_CONF^2)). The plain
            # MAGERR above propagates only the statistical (sky-limited) FLUXERR
            # and is overconfident where confusion matters (1.5-1.8x typical,
            # 10-30x in crowded fields); downstream SED fitting / photo-z should
            # consume MAGERR_TOT instead. Emitted only when FLUXERR_TOT exists
            # (compute.gpu.error_model.confusion_enabled); MAGERR is unchanged so
            # existing consumers are unaffected. Same zp_err quadrature as MAGERR.
            ferr_tot_col = f"FLUXERR_TOT_{band}_fit"
            if ferr_tot_col in cat.columns:
                ferr_tot = pd.to_numeric(cat[ferr_tot_col], errors="coerce").to_numpy(dtype=float)
                magerr_tot = np.full(len(cat), np.nan, dtype=float)
                ok_tot = ok & np.isfinite(ferr_tot) & (ferr_tot > 0)
                magerr_tot[ok_tot] = np.sqrt(
                    (K_MAG * (ferr_tot[ok_tot] / flux[ok_tot])) ** 2 + zp_err_row[ok_tot] ** 2)
                mag_updates[f"MAGERR_TOT_{band}_fit"] = magerr_tot

            # Per-source 5-sigma forced-photometry UPPER-LIMIT magnitude
            # (#28b): MAG_UL5 = zp - 2.5*log10(5 * FLUXERR_TOT). A PROPAGATED
            # per-source limit (Fisher + #27 confusion via FLUXERR_TOT, or the
            # plain statistical FLUXERR when TOT is absent), replacing the
            # vacuous header UL5_5 -- which is an analytic white-noise function
            # of SKYSIG that ignores confusion (Fable re-audit: <=3 mmag
            # identity to 5*sqrt-of-that). Defined wherever the flux ERROR is
            # finite (incl. faint / non-detections at the forced position; no
            # flux>0 requirement), and NaN for masked / never-fit rows -- so
            # those stay DISTINGUISHABLE from genuine detection limits and no
            # longer masquerade as 99-flag ULs (#28b defect 1). Same per-source
            # zp as MAG; additive column, existing consumers unaffected.
            if ferr_tot_col in cat.columns:
                ul_err = pd.to_numeric(cat[ferr_tot_col], errors="coerce").to_numpy(dtype=float)
            else:
                ul_err = ferr
            mag_ul5 = np.full(len(cat), np.nan, dtype=float)
            ok_ul = np.isfinite(ul_err) & (ul_err > 0)
            mag_ul5[ok_ul] = zp_at_all[ok_ul] - 2.5 * np.log10(5.0 * ul_err[ok_ul])
            mag_updates[f"MAG_UL5_{band}_fit"] = mag_ul5

            # R14: the 5-sigma limit from the RECOMMENDED error, FLUXERR_MARG_TOT
            # (which carries the sqrt(max(1, rchisq)) rescale that FLUXERR_TOT
            # lacks). MAG_UL5 above keeps its definition; the two differ by
            # 1.25*log10(max(1, rchisq)) on all-STAR rows (T08676: median
            # 0.36 mag, max 3.7; T08285: median 0, tail to 5.3). NaN where
            # FLUXERR_MARG_TOT is: no information, or no rchisq (island_giant
            # rows, never reliable).
            fe_mt_col = f"FLUXERR_MARG_TOT_{band}_fit"
            if fe_mt_col in cat.columns:
                fe_mt = pd.to_numeric(cat[fe_mt_col], errors="coerce").to_numpy(dtype=float)
                mag_ul5m = np.full(len(cat), np.nan, dtype=float)
                ok_ulm = np.isfinite(fe_mt) & (fe_mt > 0)
                mag_ul5m[ok_ulm] = zp_at_all[ok_ulm] - 2.5 * np.log10(5.0 * fe_mt[ok_ulm])
                mag_updates[f"MAG_UL5_MARG_{band}_fit"] = mag_ul5m

            # D2 (root-fix D): SHAPE-marginalized magnitude errors from the
            # flux-shape-marginalized FLUXERR_MARG / FLUXERR_MARG_TOT columns
            # (compute.gpu.error_model.marginal_flux_err_enabled, island mode).
            # FLUXERR_MARG folds in the flux-shape covariance the CONDITIONAL
            # FLUXERR omits; downstream SED fitting / photo-z should consume
            # MAGERR_MARG_TOT for honest extended-source flux errors. Same
            # zp_err quadrature as MAGERR/MAGERR_TOT; MAGERR is unchanged so
            # existing consumers are unaffected. Emitted only when the MARG
            # flux columns exist. (FLUXERR_MARG_TOT carries the rchisq rescale
            # that FLUXERR_TOT does not, so STAR rows differ there.)
            for _src, _dst in ((f"FLUXERR_MARG_{band}_fit", f"MAGERR_MARG_{band}_fit"),
                               (f"FLUXERR_MARG_TOT_{band}_fit",
                                f"MAGERR_MARG_TOT_{band}_fit")):
                if _src in cat.columns:
                    fe_m = pd.to_numeric(cat[_src], errors="coerce").to_numpy(dtype=float)
                    magerr_m = np.full(len(cat), np.nan, dtype=float)
                    ok_m = ok & np.isfinite(fe_m) & (fe_m > 0)
                    magerr_m[ok_m] = np.sqrt(
                        (K_MAG * (fe_m[ok_m] / flux[ok_m])) ** 2 + zp_err_row[ok_m] ** 2)
                    mag_updates[_dst] = magerr_m

        # R08 -> R06: a band whose ZP fell back is no longer usable; a fitted
        # source left with no usable band hard-fails (no_usable_band), as the
        # merge stage does for the other reasons.
        if flag_cols and "n_bands_usable" in cat.columns:
            nbu_after = _count_usable_bands(cat, flag_cols)
            cat["n_bands_usable"] = nbu_after
            if "fit_is_reliable" in cat.columns:
                rel = cat["fit_is_reliable"].fillna(False).astype(bool).to_numpy()
                demote = rel & (nbu_before > 0) & (nbu_after == 0)
                if np.any(demote):
                    cat["fit_is_reliable"] = rel & ~demote
                    fq = (cat["fit_quality_issues"].fillna("").astype(str).to_numpy()
                          if "fit_quality_issues" in cat.columns else np.full(len(cat), "", dtype=object))
                    cat["fit_quality_issues"] = np.where(
                        demote, np.where(fq == "", "no_usable_band", fq + ",no_usable_band"), fq)
                logger.info("ZP support: %d reliable row(s) lost their last usable band "
                            "(no_usable_band)", int(np.count_nonzero(demote)))

        for row in summary_rows:
            m = spatial_models.setdefault(row["band"], {"kind": "scalar",
                                                        "zp_median": float(row["zp_median"])})
            m["zp_err"] = float(row[_err_key])
            m["zp_err_fallback"] = float(row["zp_err_fallback"])
        _write_zp_models(zp_dir / "zp_model.json", spatial_models,
                         k=spatial_zp_min_per_cell, tol=ZP_SUPPORT_TOL_MAG)

        if mag_updates:
            replace_cols = [c for c in mag_updates if c in cat.columns]
            # This stage owns every MAG*_fit column: on a `tract7dt
            # compute-zp` re-run, columns an older version wrote and this one
            # does not (e.g. the all-NaN MAG(ERR)_MARG_FRAC_* before R14) go.
            stale = [c for c in cat.columns if c.startswith("MAG") and c.endswith("_fit")
                     and c not in mag_updates]
            if stale:
                logger.info("Dropping %d magnitude column(s) this version does not "
                            "write (e.g. %s)", len(stale), stale[0])
            replace_cols += stale
            if replace_cols:
                cat = cat.drop(columns=replace_cols)
            cat = pd.concat([cat, pd.DataFrame(mag_updates, index=cat.index)], axis=1)

        write_csv_parallel(cat, merged_catalog_path, workers=csv_write_workers)
        logger.info("Updated merged catalog with MAG/MAGERR columns: %s", merged_catalog_path)
    else:
        logger.warning("No bands produced ZP results; catalog unchanged")
        if passthrough:
            # In-memory handoff: still must emit the catalog to disk.
            write_csv_parallel(cat, merged_catalog_path, workers=csv_write_workers)
    _calibrated = [row["band"] for row in summary_rows]
    _without = {b: zp_missing.get(b, "no zero point") for b in bandnames
                if b not in _calibrated}
    run_status.note("zp", {"bands_calibrated": _calibrated,
                           "bands_without_zp": _without})
    for _b, _why in _without.items():
        run_status.warn(f"[{_b}] no zero point ({_why}): its MAG columns are "
                        "NaN although the band was fitted")

    # Drain the diagnostic-plot pool (submitted before the catalog writes so it
    # rendered concurrently); on a 1-core host render inline here instead.
    if plot_tasks:
        if plot_ex is not None:
            for fut in as_completed(plot_futs):
                band_done, err = fut.result()
                if err:
                    logger.warning("ZP diagnostic plot failed for %s: %s", band_done, err)
            plot_ex.shutdown()
        else:
            for task in plot_tasks:
                band_done, err = _save_zp_band_plots_worker(task)
                if err:
                    logger.warning("ZP diagnostic plot failed for %s: %s", band_done, err)
        logger.info("ZP diagnostic plots done in %.2fs (overlapped with writes)",
                    time.perf_counter() - t_plot)

    # --- Image<->Gaia frame-tie FIELD-LEVEL DIAGNOSTIC (C2 REVERTED 2026-07-11) ---
    # Optional, OFF by default, and PURELY diagnostic: it NEVER modifies POS_ERR
    # or any fit input. When zp.frame_tie_diagnostic == "measure", measure the
    # COHERENT image<->Gaia offset per field from the reliable Gaia-matched
    # calibrators' fit-vs-seed residuals (de-biased for prior shrinkage), LOG it
    # (one line) and write the sidecar "<stem>.frame_tie.json" for the record.
    # This coherent offset is a WCS / absolute-astrometry systematic of the whole
    # field, not a per-source position error, so it belongs to a downstream WCS /
    # absolute-astrometry correction -- never a per-source error-bar modifier.
    # Reuses the ZP sigma-clip / min-calibrator hygiene -- no dataset-tuned
    # constants. The sidecar is never read back into POS_ERR.
    ft_diag = str(zp_cfg.get("frame_tie_diagnostic", "off")).strip().lower()
    _ft_need = {"gaia_source_id", "fit_is_reliable", "x_pix_white", "y_pix_white",
                "x_pix_white_fit", "y_pix_white_fit", "x_pix_white_prior_frac"}
    if ft_diag == "measure" and _ft_need <= set(cat.columns):
        try:
            from astropy.io import fits as _fits
            from astropy.wcs import WCS as _WCS
            from astropy.wcs.utils import proj_plane_pixel_scales as _pps
            _wpath = Path(outputs["work_dir"]) / "wcs.fits"
            _plate = (float(np.mean(_pps(_WCS(_fits.getheader(_wpath)))) * 3600.0)
                      if _wpath.is_file() else float("nan"))
            _gid = cat["gaia_source_id"].fillna("").astype(str).str.strip()
            _xw = pd.to_numeric(cat["x_pix_white"], errors="coerce")
            _xf = pd.to_numeric(cat["x_pix_white_fit"], errors="coerce")
            _sel = ((_gid != "") & (_gid != "nan")
                    & cat["fit_is_reliable"].fillna(False).astype(bool)
                    & np.isfinite(_xw) & np.isfinite(_xf)).to_numpy()
            _c = cat.loc[_sel]
            _pfx = pd.to_numeric(_c["x_pix_white_prior_frac"], errors="coerce").to_numpy(float)
            _pfy = pd.to_numeric(_c.get("y_pix_white_prior_frac",
                                        _c["x_pix_white_prior_frac"]),
                                 errors="coerce").to_numpy(float)
            _pf = np.where(np.isfinite(_pfx) & np.isfinite(_pfy), 0.5 * (_pfx + _pfy),
                           np.where(np.isfinite(_pfx), _pfx, _pfy))
            info = estimate_frame_tie_floor_mas(
                pd.to_numeric(_c["x_pix_white"], errors="coerce").to_numpy(float),
                pd.to_numeric(_c["y_pix_white"], errors="coerce").to_numpy(float),
                pd.to_numeric(_c["x_pix_white_fit"], errors="coerce").to_numpy(float),
                pd.to_numeric(_c["y_pix_white_fit"], errors="coerce").to_numpy(float),
                _pf, _plate,
                clip_sigma=clip_sigma, clip_max_iters=clip_max_iters,
                min_calibrators=int(zp_cfg.get("gaia_min_calibrators", 10)))
            if info.get("ok"):
                logger.info(
                    "Frame-tie DIAGNOSTIC (%d/%d reliable Gaia calibrators): "
                    "coherent image<->Gaia offset=%.1f mas (max-axis %.1f mas; "
                    "bulk %.1f mas; per-star PM/centroid %.1f mas). FIELD-LEVEL "
                    "WCS/absolute-astrometry systematic -- NOT applied to POS_ERR; "
                    "logged + persisted for the record only.",
                    info["n_used"], info["n_in"], info["floor_mas"],
                    info["floor_mas_maxaxis"], info["bulk_mag_mas"],
                    info["perstar_resid_mas"])
                _fp = write_frame_tie_floor(inputs.get("input_catalog"), info)
                if _fp:
                    logger.info("Frame-tie diagnostic persisted (record only): %s", _fp)
            else:
                logger.info("Frame-tie diagnostic skipped (%s)", info.get("note"))
        except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
            logger.info("Frame-tie diagnostic failed (%s); nothing persisted", exc)

    if not summary_rows:
        raise RunContractError(
            "zp_not_computed",
            f"no band produced a zero point ({_without}); "
            f"{Path(merged_catalog_path).name} was written without magnitudes")
    return merged_catalog_path
