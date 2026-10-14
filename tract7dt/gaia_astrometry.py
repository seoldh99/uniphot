"""Gaia DR3 astrometric-error epoch propagation for position-prior widths.

The pipeline's Gaussian position prior is opt-in per source via the
``POS_ERR`` column (sigma [pixels] for both axes; see docs/input-catalog.md).
This module computes an *honest, per-source* width for Gaia-seeded rows from
Gaia DR3's own published astrometric uncertainties, instead of a fixed
constant stamped on every row:

    sigma_axis(t_obs) = sqrt( sigma_pos_2016^2 + (t_obs - 2016.0)^2 * sigma_pm^2 )

per axis [mas], where ``sigma_pos_2016`` is the Gaia DR3 position error at
the catalog reference epoch (``ra_error`` is already sigma_ra* = the
on-sky RA axis error, so no cos(dec) factor is needed) and ``sigma_pm`` is
the proper-motion error. The conservative max over the two axes is taken
(the fit applies one isotropic sigma to x and y), converted to pixels via
the image plate scale.

Policy (no dataset-tuned constants -- see the project directive): every
number here is data-derived (per-source Gaia DR3 errors, the observation
epoch from the image headers, the plate scale from the image WCS) or an
explicit user belief (``floor_mas``). Sources WITHOUT a proper-motion
solution (2-parameter, ``astrometric_params_solved == 3``, or missing PM
error columns) cannot be epoch-propagated: they get ``NaN`` (= NO prior),
*even when* ``floor_mas > 0`` -- a floor alone would fake knowledge of an
unknown epoch-propagated position. The floor only broadens (in quadrature)
widths that are genuinely known.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("tract7dt.gaia_astrometry")

#: Gaia DR3 columns required for epoch propagation (all in mas / mas yr^-1).
GAIA_POS_ERR_COLS = ("ra_error", "dec_error", "pmra_error", "pmdec_error")

#: Gaia DR3 catalog reference epoch (Julian year, TCB).
GAIA_DR3_REF_EPOCH = 2016.0


def has_gaia_pos_err_columns(df: pd.DataFrame) -> bool:
    """True if ``df`` carries all Gaia astrometric-error columns needed for
    :func:`propagate_gaia_pos_err` (``astrometric_params_solved`` is optional)."""
    return all(c in df.columns for c in GAIA_POS_ERR_COLS)


def propagate_gaia_pos_err(
    df: pd.DataFrame,
    epoch_obs_yr: float,
    plate_scale_arcsec_px: float,
    *,
    ref_epoch: float = GAIA_DR3_REF_EPOCH,
    floor_mas: float = 0.0,
) -> np.ndarray:
    """Per-source position-prior width [pixels] at the observation epoch.

    Parameters
    ----------
    df : DataFrame with Gaia DR3 columns ``ra_error``, ``dec_error`` [mas],
        ``pmra_error``, ``pmdec_error`` [mas/yr] and (optionally)
        ``astrometric_params_solved``.
    epoch_obs_yr : observation epoch as a decimal Julian year (e.g. from the
        image ``MJD-OBS``/``DATE-OBS`` headers via :func:`obs_epoch_from_headers`).
    plate_scale_arcsec_px : image plate scale [arcsec / pixel] (e.g. from the
        image WCS via ``astropy.wcs.utils.proj_plane_pixel_scales``).
    ref_epoch : catalog reference epoch (Gaia DR3: 2016.0).
    floor_mas : optional user-belief systematic floor [mas], added in
        quadrature to *propagated* widths only. It never substitutes for a
        missing PM solution (those rows stay ``NaN`` = no prior).

    Returns
    -------
    ndarray (len(df),) of POS_ERR [pixels]; ``NaN`` where the row cannot be
    propagated (missing/invalid errors, or a 2-parameter solution).
    """
    if not (np.isfinite(epoch_obs_yr) and np.isfinite(plate_scale_arcsec_px)
            and plate_scale_arcsec_px > 0):
        raise ValueError(
            "propagate_gaia_pos_err: epoch_obs_yr and plate_scale_arcsec_px "
            f"must be finite (got {epoch_obs_yr!r}, {plate_scale_arcsec_px!r})")

    n = len(df)

    def _col(name: str) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        return np.full(n, np.nan, dtype=float)

    ra_e, dec_e, pmra_e, pmdec_e = (_col(c) for c in GAIA_POS_ERR_COLS)

    dt = float(epoch_obs_yr) - float(ref_epoch)
    sig_ra = np.sqrt(ra_e ** 2 + dt ** 2 * pmra_e ** 2)
    sig_dec = np.sqrt(dec_e ** 2 + dt ** 2 * pmdec_e ** 2)
    # np.maximum propagates NaN: a missing error on EITHER axis (incl. the
    # NULL PM errors of 2-parameter sources) -> NaN -> no prior.
    sig_mas = np.maximum(sig_ra, sig_dec)

    valid = np.isfinite(sig_mas) & (sig_mas > 0)
    if "astrometric_params_solved" in df.columns:
        solved = pd.to_numeric(
            df["astrometric_params_solved"], errors="coerce").to_numpy(dtype=float)
        # 3 = 2-parameter (position-only) solution: no PM -> not propagatable
        # even if the CSV carries (meaningless) filled PM-error values.
        valid &= ~(solved == 3.0)

    if floor_mas and float(floor_mas) > 0:
        sig_mas = np.sqrt(sig_mas ** 2 + float(floor_mas) ** 2)

    out = np.where(
        valid, sig_mas / 1000.0 / float(plate_scale_arcsec_px), np.nan)
    return out.astype(float)


def obs_epoch_from_headers(headers) -> float | None:
    """Median observation epoch (decimal Julian year) from FITS headers.

    Prefers ``MJD-OBS`` and falls back to ``DATE-OBS`` per header. Returns
    ``None`` when no header yields a usable epoch (caller decides fallback).
    """
    from astropy.time import Time

    years: list[float] = []
    for hdr in headers:
        if hdr is None:
            continue
        t = None
        mjd = hdr.get("MJD-OBS")
        try:
            if mjd is not None and np.isfinite(float(mjd)):
                t = Time(float(mjd), format="mjd")
        except (TypeError, ValueError):
            t = None
        if t is None:
            date_obs = hdr.get("DATE-OBS")
            if date_obs:
                try:
                    t = Time(str(date_obs), format="isot")
                except ValueError:
                    t = None
        if t is not None:
            years.append(float(t.decimalyear))
    if not years:
        return None
    return float(np.median(years))


# ---------------------------------------------------------------------------
#  Image<->Gaia frame-tie FIELD-LEVEL DIAGNOSTIC (root-fix C2, REVERTED 2026-07-11)
# ---------------------------------------------------------------------------
#
# The epoch-propagated Gaia width above captures the star's OWN astrometric
# uncertainty (~0.5-1 mas for bright ZP calibrators) -- a KNOWN per-source input
# uncertainty, which is the honest per-source position error. Separately, there
# is a systematic offset between the Gaia sky frame (projected via the coadd WCS)
# and where sources actually sit in the coadd -- the image<->Gaia *frame-tie*.
# Measured on the T08285 bulge subfield it is a *coherent* ~72 mas (bulk, one-
# axis) offset + ~15 mas smooth spatial gradient, with only ~15 mas per-star
# scatter.
#
# IMPORTANT (why C2 was reverted): position is DATA-DEFINED and ABSOLUTE, so its
# uncertainty must be PROPAGATED from a known input error (the Gaia width) -- it
# may NOT be back-measured by comparing the fitted position to the Gaia "answer"
# and folded into a per-source error bar. That coherent offset is a property of
# the WHOLE FIELD's WCS / absolute-astrometry tie, not of any one source, and it
# is undefined for the reference-less science sources a POS_ERR floor would then
# modify. The frame-tie therefore belongs to WCS calibration / a downstream
# absolute-astrometry correction, NOT to POS_ERR.
#
# :func:`estimate_frame_tie_floor_mas` is kept ONLY as a field-level DIAGNOSTIC
# (opt-in `zp.frame_tie_diagnostic: "measure"`): it measures + logs the coherent
# offset and writes it to a sidecar for the record. It is NEVER wired into
# POS_ERR, any prior width, or any fit input. The measurement itself is honest
# (no dataset-tuned constant; the only constants are robust-estimator hygiene --
# sigma-clip kappa, min calibrator count -- reused from the ZP calibrator
# rejection; per-star PM/centroid scatter is excluded by construction as it does
# not project onto the low-order coherent basis). The function is also retained
# for possible future WCS-time (absolute-astrometry) use.


def estimate_frame_tie_floor_mas(
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    prior_frac: np.ndarray,
    plate_scale_arcsec_px: float,
    *,
    clip_sigma: float = 3.0,
    clip_max_iters: int = 10,
    min_calibrators: int = 10,
) -> dict:
    """Estimate the per-field image<->Gaia frame-tie floor [mas].

    Recovers the COHERENT frame-tie field ``F(x, y)`` from the ZP calibrators'
    fit-vs-seed residuals, de-biased for the position-prior shrinkage so it
    works even when the calibrators were fit prior-pinned (the common case with
    tight propagated widths). Model per axis::

        residual_i = (fit_i - seed_i) ~= (1 - prior_frac_i) * F(x_i, y_i)

    (a source with position prior_frac ``p`` lands at ``seed + (1-p)*(data-seed)``;
    ``data - seed`` ~= the frame-tie field). Regressing the residual on the
    de-biased design ``(1-p) * [1, x_n, y_n]`` recovers ``F`` = bulk + linear
    spatial term. The floor is the robust RMS of ``F`` over the (sigma-clipped)
    calibrators, per axis, combined. Per-star proper-motion / centroid scatter
    is zero-mean and does not project onto the low-order coherent basis, so it is
    excluded (and a well-tied frame returns ~0 -- the double-count guard).

    Parameters
    ----------
    seed_x, seed_y : Gaia-seed positions [pixels] (catalog RA/DEC via the WCS).
    fit_x, fit_y : fitted positions [pixels].
    prior_frac : per-source position prior fraction in ``[0, 1)`` (the ratio
        ``prior_inv2 / (H + prior_inv2)`` the fit already discloses); a
        prior-free row is 0. Use ``0.5*(x_prior_frac + y_prior_frac)`` when
        both axes are available.
    plate_scale_arcsec_px : image plate scale [arcsec / pixel].
    clip_sigma, clip_max_iters, min_calibrators : robust-estimator hygiene
        (reuse the ZP calibrator-rejection values; not tuned to any dataset).

    Returns
    -------
    dict with ``floor_mas`` (the recommended isotropic per-axis floor = coherent
    RMS/axis), ``floor_px`` (converted with the plate scale), and diagnostics
    (``floor_mas_maxaxis``, ``coh_x``/``coh_y``, ``bulk_mag_mas``,
    ``perstar_resid_mas``, ``n_used``, ``n_in``, ``ok``).
    """
    seed_x = np.asarray(seed_x, dtype=float)
    seed_y = np.asarray(seed_y, dtype=float)
    fit_x = np.asarray(fit_x, dtype=float)
    fit_y = np.asarray(fit_y, dtype=float)
    p = np.clip(np.asarray(prior_frac, dtype=float), 0.0, 0.999)
    if not (np.isfinite(plate_scale_arcsec_px) and plate_scale_arcsec_px > 0):
        return {"floor_mas": 0.0, "floor_px": 0.0, "ok": False,
                "note": "invalid plate scale", "n_in": int(len(seed_x))}

    rx = (fit_x - seed_x) * plate_scale_arcsec_px * 1000.0
    ry = (fit_y - seed_y) * plate_scale_arcsec_px * 1000.0
    good = (np.isfinite(rx) & np.isfinite(ry) & np.isfinite(p)
            & np.isfinite(seed_x) & np.isfinite(seed_y))
    rx, ry = rx[good], ry[good]
    sx, sy = seed_x[good], seed_y[good]
    w = 1.0 - p[good]                       # data-dominance weight
    n_in = int(len(rx))
    if n_in < int(min_calibrators):
        return {"floor_mas": 0.0, "floor_px": 0.0, "ok": False,
                "note": "too few calibrators", "n_in": n_in}

    def _nmad(v):
        v = v[np.isfinite(v)]
        return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) >= 3 else 0.0

    sx_s = float(sx.std()) or 1.0
    sy_s = float(sy.std()) or 1.0
    xn = (sx - sx.mean()) / sx_s
    yn = (sy - sy.mean()) / sy_s
    basis = np.vstack([np.ones_like(xn), xn, yn]).T     # (n, 3): bulk + linear
    design = basis * w[:, None]                         # de-biased for shrinkage

    keep = np.ones(n_in, dtype=bool)
    cx = cy = None
    for _ in range(max(1, int(clip_max_iters))):
        if keep.sum() < int(min_calibrators):
            break
        cx = np.linalg.lstsq(design[keep], rx[keep], rcond=None)[0]
        cy = np.linalg.lstsq(design[keep], ry[keep], rcond=None)[0]
        resx = rx - design @ cx
        resy = ry - design @ cy
        rr = np.hypot(resx, resy)
        med = float(np.median(rr[keep]))
        scat = _nmad(rr[keep])
        new_keep = rr < (med + float(clip_sigma) * (scat + 1e-9))
        if int(new_keep.sum()) == int(keep.sum()):
            keep = new_keep
            break
        keep = new_keep
    if cx is None:
        return {"floor_mas": 0.0, "floor_px": 0.0, "ok": False,
                "note": "regression failed", "n_in": n_in}

    # Coherent frame-tie field F(x, y) = basis @ coefs (WITHOUT the (1-p) shrink).
    fld_x = basis @ cx
    fld_y = basis @ cy
    k = keep if keep.sum() >= int(min_calibrators) else np.ones(n_in, bool)
    coh_x = float(np.sqrt(np.median(fld_x[k] ** 2)))
    coh_y = float(np.sqrt(np.median(fld_y[k] ** 2)))
    coh_rms_axis = float(np.sqrt((coh_x ** 2 + coh_y ** 2) / 2.0))
    coh_max_axis = float(max(coh_x, coh_y))
    bulk_mag = float(np.hypot(cx[0], cy[0]))
    perstar = float(np.hypot(_nmad((rx - design @ cx)[k]),
                             _nmad((ry - design @ cy)[k])) / np.sqrt(2))
    floor_mas = max(0.0, coh_rms_axis)
    return {
        "floor_mas": round(floor_mas, 3),
        "floor_px": round(floor_mas / 1000.0 / float(plate_scale_arcsec_px), 6),
        "floor_mas_maxaxis": round(coh_max_axis, 3),
        "coh_x_mas": round(coh_x, 3), "coh_y_mas": round(coh_y, 3),
        "bulk_mag_mas": round(bulk_mag, 3),
        "perstar_resid_mas": round(perstar, 3),
        "n_used": int(k.sum()), "n_in": n_in,
        "plate_scale_arcsec_px": float(plate_scale_arcsec_px),
        "ok": True,
    }


def _frame_tie_json_path(input_catalog_path) -> Path:
    """Per-field persistence path for the measured frame-tie DIAGNOSTIC.

    Keyed to the input catalog (a field's calibration product). Under
    ``zp.frame_tie_diagnostic: "measure"`` compute_zp writes the measured
    coherent offset here FOR THE RECORD only. The sidecar is NEVER read back
    into POS_ERR (C2 reverted 2026-07-11).
    """
    p = Path(input_catalog_path)
    return p.with_name(p.stem + ".frame_tie.json")


def write_frame_tie_floor(input_catalog_path, info: dict) -> Path | None:
    """Persist the measured frame-tie diagnostic next to the input catalog
    (record only; never consumed by POS_ERR)."""
    try:
        out = _frame_tie_json_path(input_catalog_path)
        with open(out, "w") as f:
            json.dump(info, f, indent=2)
        return out
    except Exception as exc:  # noqa: BLE001 - best-effort persistence
        logger.info("frame-tie: could not persist floor (%s)", exc)
        return None


def read_frame_tie_floor(input_catalog_path) -> dict | None:
    """Read the persisted per-field frame-tie diagnostic, if present.

    NOTE (C2 reverted 2026-07-11): retained for inspection / possible future
    WCS-time (absolute-astrometry) use only. It is NOT wired into POS_ERR or any
    fit input by the default pipeline path.
    """
    try:
        p = _frame_tie_json_path(input_catalog_path)
        if not p.is_file():
            return None
        with open(p) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.info("frame-tie: could not read persisted floor (%s)", exc)
        return None


def resolve_frame_tie_floor_px(
    setting,
    *,
    input_catalog_path=None,
    plate_scale_arcsec_px: float | None = None,
) -> tuple[float, str]:
    """Resolve a frame-tie floor [pixels] from a legacy setting string.

    DEPRECATED / UNWIRED (C2 reverted 2026-07-11). This helper used to turn the
    old ``zp.gaia_pos_err_frame_tie`` setting into a POS_ERR floor; POS_ERR is
    no longer modified by any frame-tie term (position is data-defined and its
    error is propagated from the known Gaia width, never back-measured against
    Gaia). It is retained only for reference / potential future WCS-time use and
    is NOT called by the default pipeline path.

    * ``"off"`` -> ``(0.0, "off")``.
    * a number (mas) -> value converted to pixels.
    * ``"auto"`` -> the per-field persisted value, else ``(0.0, "auto-pending")``.
    """
    s = setting
    if isinstance(s, str) and s.strip().lower() == "off":
        return 0.0, "off"
    if not (plate_scale_arcsec_px and np.isfinite(plate_scale_arcsec_px)
            and plate_scale_arcsec_px > 0):
        return 0.0, "no-plate-scale"
    # explicit numeric mas value
    if not isinstance(s, str):
        try:
            mas = float(s)
        except (TypeError, ValueError):
            return 0.0, "invalid"
        return (max(0.0, mas) / 1000.0 / float(plate_scale_arcsec_px), "fixed")
    if s.strip().lower() == "auto":
        info = read_frame_tie_floor(input_catalog_path) if input_catalog_path else None
        if info and info.get("ok") and float(info.get("floor_mas", 0.0)) > 0:
            mas = float(info["floor_mas"])
            return (mas / 1000.0 / float(plate_scale_arcsec_px), "auto")
        return 0.0, "auto-pending"
    return 0.0, "unknown"
