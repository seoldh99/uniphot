"""PSFEx-style direct-from-stars 2D-varying polynomial PSF model.

A single per-band fit that goes DIRECTLY from individual SEP-detected stars to
the polynomial PSF, faithful to PSFEx 3.24.2 (Bertin 2011) algorithms:

- ``compute_fwhmrange`` (PSFEx ``sample.c``): sliding-window mode
  finder on sorted FWHMs.
- training stars (M2 unit a.3, design_v2.md 1.5; replaces PSFEx's
  ``read_samples`` cuts): SEP detections paired with input rows that have a
  Gaia identity and pass the Gaia quality, S/N >= min_snr, and only data
  properties as hard exclusions (saturated or bad pixel in the cutout, peak
  at the linear limit, a brighter peak in the box); cleanliness is a
  continuous star weight (``psf0_star_noise``), not a cut.
- ``psf_make`` + ``psf_makeresi`` + ``psf_clean``: outer iterative
  outlier rejection on the polynomial fit (kappa-sigma clipping on the
  per-star sqrt(chi2) distribution).

Sub-pixel star centering supports two algorithms:

- ``centering_method="gw_centroid"`` (default, recommended):
  vectorized iterative Gaussian-weighted centroid (a la SExtractor
  WIN moments / Stetson centroider). Runs all stars in a single numpy
  broadcast; ~100x faster than ``"epsfbuilder"`` per band, with PSF
  model differences below 0.1% peak amplitude (measured with the old
  Lanczos registration). Pure numpy/scipy, no JAX.
- ``centering_method="epsfbuilder"``: photutils' ``EPSFBuilder``
  (Anderson & King 2000 algorithm) - kept for parity testing /
  reproducing pre-Apr-2026 outputs. Slow because we discard the
  averaged ePSF and only use the per-star fitted centers, so the
  ``_build_epsf_step`` work and 30 outer iters of per-star scipy
  least_squares are wasted.

Both register the stamp on its centre with :func:`fourier_shift` (M2 unit
a.3, design v2.5 3rd ed. 5.5; Lanczos-3 sharpened the mean registered
stamp by +0.23 % at the peak for FWHM 3.7-4.5 px).

The output is the same :class:`PsfPolynomial` dataclass that
``run_grid_gpu_fit`` already consumes -- this is a drop-in replacement
for the per-cell + polynomial PSF pipeline.

See :file:`docs/grid-gpu-roadmap.md` Phase 8 for design rationale.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sep

from ...sep_limits import apply_sep_limits
from .psf_polynomial import PsfPolynomial
from .bg_polynomial import terms_for_order, n_terms_for_order


logger = logging.getLogger("tract7dt.fit.grid_gpu.psf_from_stars")


# ---------------------------------------------------------------------------
# Diagnostics container
# ---------------------------------------------------------------------------


@dataclass
class PsfFromStarsResult:
    """One band's polynomial PSF + the bookkeeping needed for diagnostics."""
    band: str
    psf_polynomial: PsfPolynomial          # the fitted polynomial PSF model
    polynomial_order: int                  # order used (1, 2, or 3)
    n_stars_detected: int                  # raw SEP detections passing edge+sat cut
    n_stars_after_quality: int             # after every training-star rule (select_psf_stars)
    n_stars_after_gaia: int                # paired with a trainable input row (after_identity; summary.json
                                           #   "n_stars_after_gaia" -- before M2 a.3: after the quality cuts)
    n_stars_used_in_fit: int               # final after outlier rejection
    fwhm_mode: float                       # PSFEx auto-mode FWHM (pix)
    fwhm_min: float                        # auto-mode FWHM lower bound
    fwhm_max: float                        # auto-mode FWHM upper bound
    star_x: np.ndarray                     # (n_used,) full-image x coords
    star_y: np.ndarray                     # (n_used,) full-image y coords
    star_chi2: np.ndarray                  # (n_used,) per-star chi^2 from final fit
    star_flux: np.ndarray                  # (n_used,) sep flux estimate
    star_fwhm: np.ndarray                  # (n_used,) sep FWHM estimate
    star_ellip: np.ndarray                 # (n_used,) sep ellipticity (1 - b/a)
    star_origin: np.ndarray                # (n_used,) "gaia" or "star" (TrainingRows.origin)
    star_centered_stamps: np.ndarray       # (n_used, psf_size, psf_size) for diag
    n_outer_iters: int                     # PSFEx-style outlier-rejection iters used
    final_chi2_mean: float
    final_chi2_max: float
    timing_sec: float
    # --- crowding-robust path diagnostics (defaults keep legacy path valid) ---
    star_local_bkg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_neighbor_pix_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_masked_pix_frac: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_core_norm_flux: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_weighted_chi2: np.ndarray = field(default_factory=lambda: np.zeros(0))
    star_n_neighbors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    psf_normalization_used: str = "sum"
    r_core_pix: float = float("nan")
    median_neighbor_pix_frac: float = float("nan")
    central_resid_frac: float = float("nan")
    central_monopole_frac: float = float("nan")
    central_dipole_frac: float = float("nan")
    n_singular_px: int = 0
    min_neff_px: float = float("nan")


# ---------------------------------------------------------------------------
# Step 1: SEP star detection + measurement
# ---------------------------------------------------------------------------


def detect_sep_objects(
    image: np.ndarray, sigma: np.ndarray, bad_mask: np.ndarray,
    *,
    thresh_sigma: float = 10.0,
    minarea: int = 25,
    deblend_nthresh: int = 32,
    deblend_cont: float = 0.005,
):
    """Wrap ``sep.extract`` for tract7dt-style detection. Returns the
    raw structured array from sep with fields: x, y, a, b, theta,
    peak, flux, npix, fluxerr, etc. (empty on a SEP error, logged).
    """
    img = np.ascontiguousarray(image, dtype=np.float32)
    sig = np.ascontiguousarray(sigma, dtype=np.float32)
    bad = np.ascontiguousarray(bad_mask.astype(bool), dtype=np.bool_)
    empty = np.zeros(0, dtype=np.dtype([("x", "f8"), ("y", "f8")]))
    apply_sep_limits(sep)
    try:
        out = sep.extract(
            img, thresh=float(thresh_sigma), err=sig, mask=bad,
            minarea=int(minarea),
            deblend_nthresh=int(deblend_nthresh),
            deblend_cont=float(deblend_cont),
            clean=True, clean_param=1.0,
        )
        objs = out
        if objs is None:
            return empty
        return objs
    except Exception as exc:
        logger.warning("sep.extract failed: %s", exc)
        return empty


def _objs_with_quality_metrics(
    objs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-object (x, y, fwhm, ellipticity, snr) arrays."""
    if len(objs) == 0:
        z = np.zeros(0, dtype=np.float64)
        return z, z, z, z, z
    x = np.asarray(objs["x"], dtype=np.float64)
    y = np.asarray(objs["y"], dtype=np.float64)
    # FWHM: 2.355 * sigma where sigma = sqrt((a^2 + b^2) / 2) for an
    # elliptical Gaussian. SEP's a, b are the rms semi-major/minor axes.
    a = np.asarray(objs["a"], dtype=np.float64)
    b = np.asarray(objs["b"], dtype=np.float64)
    fwhm = 2.354820045 * np.sqrt(0.5 * (a * a + b * b))
    # Ellipticity: PSFEx uses (a-b)/(a+b); SEP follows same convention.
    ab_sum = np.maximum(a + b, 1e-30)
    ellip = (a - b) / ab_sum
    # SNR: flux / fluxerr if available, else flux / sqrt(npix * background_var).
    if "flux" in objs.dtype.names and "fluxerr" in objs.dtype.names:
        flux = np.asarray(objs["flux"], dtype=np.float64)
        fluxerr = np.asarray(objs["fluxerr"], dtype=np.float64)
        snr = np.where(fluxerr > 0, flux / fluxerr, np.inf)
    elif "flux" in objs.dtype.names and "npix" in objs.dtype.names:
        flux = np.asarray(objs["flux"], dtype=np.float64)
        snr = np.full_like(flux, np.inf)
    else:
        snr = np.full_like(x, np.inf)
    return x, y, fwhm, ellip, snr


# ---------------------------------------------------------------------------
# Step 2: PSFEx-style auto FWHM mode finder (translates sample.c)
# ---------------------------------------------------------------------------


def compute_fwhmrange_psfex(
    fwhm: np.ndarray, sample_variability: float,
    fwhm_floor: float = 1.5, fwhm_ceiling: float = 30.0,
) -> tuple[float, float, float]:
    """Faithful translation of PSFEx ``compute_fwhmrange`` from
    :file:`src/sample.c` (lines 311-374).

    Algorithm:

    1. Sort FWHMs.
    2. Sliding window of width nw = nfwhm/4: find the window where
       ``(f_t2 - f_t) / ((f_t2 + f_t)/2)`` is smallest -> mode is the
       midpoint of that window.
    3. Iteratively halve nw until variance < sample_variability/2.
    4. ``fwhm_min = mode / sqrt(1 + sample_variability)``
       ``fwhm_max = mode * sqrt(1 + sample_variability)``
    5. Clip to [fwhm_floor, fwhm_ceiling].

    Returns ``(mode, fwhm_min, fwhm_max)``.
    """
    fwhm = np.asarray(fwhm, dtype=np.float64)
    fwhm = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
    nfwhm = len(fwhm)
    if nfwhm == 0:
        return float("nan"), float(fwhm_floor), float(fwhm_ceiling)
    if nfwhm == 1:
        return float(fwhm[0]), float(fwhm_floor), float(fwhm_ceiling)

    fwhm_sorted = np.sort(fwhm)
    BIG = 1e30
    nw = max(1, nfwhm // 4)
    var = BIG
    fmin = float(fwhm_sorted[nfwhm // 2])  # fallback = median
    target_var = float(sample_variability) / 2.0

    while var > target_var and nw > 0:
        dfmin = BIG
        fmin_iter = 0.0
        # For all windows of width nw, find min relative spread:
        #   (f[i+nw] - f[i]) / ((f[i+nw] + f[i])/2)
        for i in range(nfwhm - nw):
            f1 = fwhm_sorted[i]
            f2 = fwhm_sorted[i + nw]
            df = f2 - f1
            f = 0.5 * (f1 + f2)
            # PSFEx uses: df * fmin < dfmin * f  (cross-multiplied to
            # avoid div by zero). Replicate exactly.
            if df * (fmin_iter if fmin_iter > 0 else 1e-30) < dfmin * f:
                dfmin = df
                fmin_iter = f
        if fmin_iter <= 0.0:
            break
        var = dfmin / fmin_iter
        if var <= target_var:
            fmin = fmin_iter
            break
        fmin = fmin_iter
        nw //= 2
        if nw < 1:
            break

    sqrt_factor = float(np.sqrt(sample_variability + 1.0))
    fwhm_min = fmin / sqrt_factor if sqrt_factor > 0 else 0.0
    fwhm_max = fmin * sqrt_factor
    fwhm_min = float(max(fwhm_min, fwhm_floor))
    fwhm_max = float(min(fwhm_max, fwhm_ceiling))
    return float(fmin), fwhm_min, fwhm_max


# ---------------------------------------------------------------------------
# Step 3: PSF training stars (M2 unit a.3; design_v2.md 1.5, v2.4 D6/D8/D9-4)
# ---------------------------------------------------------------------------

#: The input column that gives a row its Gaia identity (design 1.5 v2.4, D8: an explicit
#: string column filled by the adapter; the identity is never guessed from the ID's shape).
#: Not the lower-case ``gaia_source_id`` that the ZP augment writes from a nearest-XP match
#: within 3 arcsec (R34): that match also lands on planted and blank rows.
GAIA_ID_COLUMN = "GAIA_SOURCE_ID"

#: Pairing radius of a SEP detection and an input row, in units of the FWHM mode: an offset
#: below half the FWHM is the same point source (the Gaia position at the image epoch and a
#: S/N >= 20 centroid are good to a few tenths of a pixel), and two sources closer than that
#: are not resolved, so a second row inside the radius makes the identity ambiguous.
TRAINING_MATCH_RADIUS_FWHM = 0.5

#: Neighbours of a training star whose centre lies farther than this many Gaussian sigmas
#: (Chebyshev) from every pixel of its box add exactly 0 to the iteration-0 neighbour light in
#: float64 (exp(-39^2/2) = exp(-760) underflows), so leaving them out changes nothing.
GAUSS_REACH_SIGMA = 39.0

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass(frozen=True)
class TrainingRows:
    """Input rows that may train the PSF of one image (design 1.5).

    index: integer positions in the catalogue of the rows with a finite pixel position;
    x, y: their pixel positions in this image's work frame (0-based, SEP convention);
    trainable: may the row be a training star; origin: "gaia" (Gaia identity rows that
    pass the Gaia quality) or "star" (no Gaia identity in the list: TYPE == STAR rows);
    note: one run.log line.
    """
    index: np.ndarray
    x: np.ndarray
    y: np.ndarray
    trainable: np.ndarray
    origin: str
    note: str


def _has_value(col) -> np.ndarray:
    s = col.astype("string").str.strip()
    empty = s.isin(["", "nan", "NaN", "None", "<NA>"]).fillna(False)
    return (s.notna() & ~empty).to_numpy(dtype=bool)


def psf_training_rows(catalog, world_to_pixel, *, max_ruwe: float) -> TrainingRows:
    """The rows of ``catalog`` that may train the PSF of one image (design 1.5, v2.4 D8).

    A row with a Gaia identity (column :data:`GAIA_ID_COLUMN`) may train when it passes the
    Gaia quality: input column ``RUWE`` <= ``max_ruwe`` and, if the column exists,
    ``NON_SINGLE_STAR`` == 0 (an unknown value fails).  Its TYPE does not matter.  Planted
    truth rows, blank spots and phantoms carry no Gaia identity, so they never train.
    A list without any Gaia identity falls back to the TYPE == STAR rows (design 1.5); the
    caller must then flag planted rows itself, which the note says.
    Rows with a Gaia identity but no ``RUWE`` column raise ValueError: the quality is never
    skipped (D8 (c)); reading the engine's Gaia quality cache instead is unit 0.5's.
    ``world_to_pixel(ra, dec) -> (x, y)`` maps to this image's 0-based pixel frame.
    """
    import pandas as pd

    cols = {str(c).strip().lower(): c for c in catalog.columns}
    ra_c, dec_c, type_c = cols.get("ra"), cols.get("dec"), cols.get("type")
    if ra_c is None or dec_c is None:
        raise ValueError("PSF training rows: the input catalogue has no RA/DEC columns")
    ra = pd.to_numeric(catalog[ra_c], errors="coerce").to_numpy(dtype=np.float64)
    dec = pd.to_numeric(catalog[dec_c], errors="coerce").to_numpy(dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(ra) & np.isfinite(dec))
    if len(idx):
        x, y = world_to_pixel(ra[idx], dec[idx])
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    else:
        x = y = np.zeros(0, dtype=np.float64)
    fin = np.isfinite(x) & np.isfinite(y)
    idx, x, y = idx[fin], x[fin], y[fin]
    sub = catalog.iloc[idx]
    ident = (_has_value(sub[GAIA_ID_COLUMN]) if GAIA_ID_COLUMN in catalog.columns
             else np.zeros(len(idx), dtype=bool))
    if ident.any():
        if "RUWE" not in catalog.columns:
            raise ValueError(
                f"PSF training rows: {int(ident.sum())} rows have a Gaia identity "
                f"({GAIA_ID_COLUMN}) but the input list has no RUWE column; the Gaia quality "
                "of a PSF training star is never skipped (design 1.5, D8). Add RUWE (and "
                "NON_SINGLE_STAR) to the input list")
        ruwe = pd.to_numeric(sub["RUWE"], errors="coerce").to_numpy(dtype=np.float64)
        good = np.isfinite(ruwe) & (ruwe <= float(max_ruwe))
        rule = f"RUWE <= {float(max_ruwe):g}"
        if "NON_SINGLE_STAR" in catalog.columns:
            nss = pd.to_numeric(sub["NON_SINGLE_STAR"], errors="coerce").to_numpy(dtype=np.float64)
            good &= nss == 0.0
            rule += " and NON_SINGLE_STAR = 0"
        trainable = ident & good
        note = (f"PSF training rows: {int(ident.sum())} with a Gaia identity ({GAIA_ID_COLUMN}), "
                f"{int(trainable.sum())} pass {rule}")
        return TrainingRows(idx, x, y, trainable, "gaia", note)
    if type_c is None:
        raise ValueError(
            f"PSF training rows: the input list has neither a Gaia identity ({GAIA_ID_COLUMN}) "
            "nor a TYPE column, so no row can train the PSF (design 1.5)")
    trainable = (sub[type_c].astype("string").str.strip().str.upper() == "STAR") \
        .fillna(False).to_numpy(dtype=bool)
    note = (f"PSF training rows: no Gaia identity in the input list (no {GAIA_ID_COLUMN} value): "
            f"design 1.5 fallback to the {int(trainable.sum())} TYPE == STAR rows -- planted or "
            "truth rows in the list are NOT kept out of the PSF training by the engine (flag "
            "them in the list)")
    return TrainingRows(idx, x, y, trainable, "star", note)


def pair_detections_with_rows(det_x, det_y, row_x, row_y, radius: float) -> np.ndarray:
    """For each detection, the row it is paired with, or -1.

    A detection d and a row r pair when r is the only row within ``radius`` of d and d is
    the only detection within ``radius`` of r (one-to-one; ambiguous pairs stay unpaired).
    """
    from scipy.spatial import cKDTree

    det_x = np.asarray(det_x, dtype=np.float64)
    det_y = np.asarray(det_y, dtype=np.float64)
    out = np.full(len(det_x), -1, dtype=np.int64)
    if len(det_x) == 0 or len(row_x) == 0:
        return out
    dxy = np.column_stack([det_x, det_y])
    rxy = np.column_stack([np.asarray(row_x, dtype=np.float64), np.asarray(row_y, dtype=np.float64)])
    rows = cKDTree(rxy)
    n_near = rows.query_ball_point(dxy, r=float(radius), return_length=True)
    one = np.flatnonzero(n_near == 1)
    if len(one) == 0:
        return out
    _, nn = rows.query(dxy[one], k=1)
    back = cKDTree(dxy).query_ball_point(rxy[nn], r=float(radius), return_length=True)
    ok = back == 1
    out[one[ok]] = nn[ok]
    return out


def _any_in_box(mask: np.ndarray, ix, iy, half: int, which: np.ndarray) -> np.ndarray:
    """For the stars in ``which``: does ``mask`` hold any True pixel in the box of half-size
    ``half`` centred on (ix, iy)?  (Boxes are clipped to the image.)"""
    H, W = mask.shape
    hit = np.zeros(len(ix), dtype=bool)
    for k in np.flatnonzero(which):
        y0, y1 = max(0, iy[k] - half), min(H, iy[k] + half + 1)
        x0, x1 = max(0, ix[k] - half), min(W, ix[k] + half + 1)
        hit[k] = bool(mask[y0:y1, x0:x1].any())
    return hit


def _brighter_peak_in_box(objs: np.ndarray, flux: np.ndarray | None, ix, iy, half: int,
                          which: np.ndarray) -> np.ndarray:
    """For the stars in ``which``: is the peak pixel of a brighter detection (larger SEP flux)
    inside the star's box of half-size ``half``?  Every detection counts as a neighbour.
    ``flux`` None: any other detection's peak counts (the isolation test)."""
    from scipy.spatial import cKDTree

    names = objs.dtype.names or ()
    if "xpeak" in names and "ypeak" in names:
        px = np.asarray(objs["xpeak"], dtype=np.float64)
        py = np.asarray(objs["ypeak"], dtype=np.float64)
    else:
        px = np.rint(np.asarray(objs["x"], dtype=np.float64))
        py = np.rint(np.asarray(objs["y"], dtype=np.float64))
    hit = np.zeros(len(ix), dtype=bool)
    sel = np.flatnonzero(which)
    if len(sel) == 0:
        return hit
    near = cKDTree(np.column_stack([px, py])).query_ball_point(
        np.column_stack([ix[sel], iy[sel]]).astype(np.float64), r=float(half), p=np.inf)
    for k, js in zip(sel, near):
        hit[k] = any(j != k and (flux is None or flux[j] > flux[k]) for j in js)
    return hit


def isolated_in_box(objs: np.ndarray, det_index, half: int = 27) -> np.ndarray:
    """For the detections ``det_index``: no other detection's peak pixel lies inside the box of
    half-size ``half`` centred on the detection's (rint x, rint y) (design v2.5 3rd ed. 5.3)."""
    det_index = np.asarray(det_index, dtype=np.int64)
    n = len(objs)
    ix = np.rint(np.asarray(objs["x"], dtype=np.float64)).astype(np.int64)
    iy = np.rint(np.asarray(objs["y"], dtype=np.float64)).astype(np.int64)
    which = np.zeros(n, dtype=bool)
    which[det_index] = True
    return ~_brighter_peak_in_box(objs, None, ix, iy, int(half), which)[det_index]


def sep_auto_flux(image: np.ndarray, objs: np.ndarray, bad_mask: np.ndarray | None = None,
                  *, return_radius: bool = False):
    """SEP's FLUX_AUTO recipe (SExtractor PHOT_AUTOPARAMS 2.5, 3.5) for every detection.

    Kron radius from the first moment inside 6 x (a, b); flux in the ellipse 2.5 x the Kron radius,
    or in a circle of radius 1.75 px when that ellipse would be smaller (SEP documentation).  The
    neighbour flux of the iteration-0 noise (design v2.5 3rd ed. 5.3).  A detection whose Kron
    radius is not finite keeps its isophotal flux (radius nan).  ``return_radius``: also the radius
    of the circle with the aperture's area (2.5 r_Kron sqrt(a b), or 1.75), for
    :func:`growth_corrected_flux` (3rd ed. 6.3).
    """
    if len(objs) == 0:
        z = np.zeros(0, dtype=np.float64)
        return (z, z.copy()) if return_radius else z
    data = np.ascontiguousarray(image, dtype=np.float64)
    mask = None if bad_mask is None else np.ascontiguousarray(bad_mask, dtype=np.bool_)
    x = np.asarray(objs["x"], dtype=np.float64)
    y = np.asarray(objs["y"], dtype=np.float64)
    a = np.asarray(objs["a"], dtype=np.float64)
    b = np.asarray(objs["b"], dtype=np.float64)
    th = np.asarray(objs["theta"], dtype=np.float64)
    iso = np.asarray(objs["flux"], dtype=np.float64)
    kron, _ = sep.kron_radius(data, x, y, a, b, th, 6.0, mask=mask)
    kron = np.asarray(kron, dtype=np.float64)
    ok = np.isfinite(kron) & (kron > 0)
    flux = iso.copy()
    if ok.any():
        f, _, _ = sep.sum_ellipse(data, x[ok], y[ok], a[ok], b[ok], th[ok], 2.5 * kron[ok],
                                  subpix=1, mask=mask)
        flux[ok] = f
    r_min = 1.75
    circ = ok & (kron * np.sqrt(a * b) < r_min)
    if circ.any():
        f, _, _ = sep.sum_circle(data, x[circ], y[circ], r_min, subpix=1, mask=mask)
        flux[circ] = f
    if not return_radius:
        return flux
    radius = np.full(len(flux), np.nan)
    radius[ok] = 2.5 * kron[ok] * np.sqrt(a[ok] * b[ok])
    radius[circ] = r_min
    return flux, radius


#: Sub-pixel phases per pixel of the iteration-0 profile table (prereg_a.3_r5.md (ga): a 16 x 16 grid of
#: phases, 17 x 17 directly integrated phase stamps, a source = bilinear blend of its four nearest).
PROFILE_PHASES = 16
#: The cosine end taper of the profile starts at this share of its support radius R (prereg_a.3_r5.md (ga)).
PROFILE_TAPER_START = 0.8
#: The support radius R is where the profile falls to this share of its peak: for a Moffat,
#: (1 + R^2 / alpha^2)^(-beta) = 1e-5, i.e. R = alpha sqrt(10^(5 / beta) - 1) (prereg_a.3_r5.md (ga)).
PROFILE_SUPPORT_DROP = 1e-5
#: Gauss-Legendre points per pixel side of the pixel-integrated profile (3 x 3, prereg_a.3_r5.md (ga)).
PROFILE_GL_ORDER = 3
_GL_T, _GL_W = np.polynomial.legendre.leggauss(24)        # per radius piece (normalisation quadrature)


def _gl_pixel(order: int):
    t, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * t, 0.5 * w


def _pixel_integrate(func, dx: float, dy: float, size: int, order: int = PROFILE_GL_ORDER) -> np.ndarray:
    """out[j, i] = integral of func(u, v) over pixel (i, j) of a size x size box (Gauss-Legendre order x order per
    pixel); (u, v) are offsets from the source, which sits (dx, dy) from the box's centre pixel."""
    t, w = _gl_pixel(order)
    c = np.arange(size, dtype=np.float64) - size // 2
    u = (c[:, None] + t[None, :] - float(dx)).ravel()
    v = (c[:, None] + t[None, :] - float(dy)).ravel()
    f = np.asarray(func(u[None, :], v[:, None]), dtype=np.float64).reshape(size, order, size, order)
    return np.tensordot(np.tensordot(f, w, axes=([3], [0])), w, axes=([1], [0]))


def _cosine_taper(r, r_sup: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    r0 = PROFILE_TAPER_START * r_sup
    return np.where(r <= r0, 1.0, np.where(r >= r_sup, 0.0,
                                           0.5 * (1.0 + np.cos(np.pi * (r - r0) / (r_sup - r0)))))


def _moffat_alpha(fwhm: float, beta: float) -> float:
    return float(fwhm) / (2.0 * np.sqrt(2.0 ** (1.0 / float(beta)) - 1.0))


def _moffat_raw(u, v, fwhm: float, beta: float):
    """Unit-integral Moffat (no support limit): (beta - 1) / (pi alpha^2) (1 + r^2 / alpha^2)^(-beta)."""
    a2 = _moffat_alpha(fwhm, beta) ** 2
    return (float(beta) - 1.0) / (np.pi * a2) * (1.0 + (u * u + v * v) / a2) ** (-float(beta))


class _TabulatedProfile:
    """A circular profile g(r) cut at its support radius ``r_sup`` with the cosine end taper, scaled to a unit
    integral over the plane, and its 17 x 17 table of pixel-integrated phase stamps (prereg_a.3_r5.md (ga)):
    tab[k, i] = the profile of a source at (i / P, k / P) px from the centre pixel of a (2 m + 1)^2 stamp,
    m = ceil(r_sup) + 2, integrated over each pixel by Gauss-Legendre 3 x 3 (no Fourier shift: no ringing).
    ``peak`` = phi_max of prereg_a.3_r5d.md 1.2 (the largest pixel of the phase-0 stamp).  Subclasses give
    ``fwhm`` and ``_g(r2)`` (r2 = r^2)."""

    def _build(self, r_sup: float) -> None:
        self.r_sup = float(r_sup)
        self.r_cut = int(np.ceil(self.r_sup))
        self.m = self.r_cut + 2
        self.norm = 1.0
        # unit plane integral: 24-point Gauss-Legendre on unit pieces of [0, 0.8 R] and [0.8 R, R] (the taper's
        # second derivative jumps at 0.8 R and R, so the pieces end there)
        r0 = PROFILE_TAPER_START * self.r_sup
        edges = np.unique(np.concatenate([np.arange(0.0, r0, 1.0), [r0], np.arange(np.ceil(r0), self.r_sup, 1.0),
                                          [self.r_sup]]))
        tot = 0.0
        for a, b in zip(edges[:-1], edges[1:]):
            r = 0.5 * (b - a) * _GL_T + 0.5 * (a + b)
            tot += 0.5 * (b - a) * float(np.sum(_GL_W * 2.0 * np.pi * r * self._radial(r)))
        self.norm = 1.0 / tot
        p = PROFILE_PHASES
        size = 2 * self.m + 1
        self.tab = np.empty((p + 1, p + 1, size, size))
        for k in range(p + 1):
            for i in range(p + 1):
                self.tab[k, i] = _pixel_integrate(self.func, i / p, k / p, size)
        self.peak = float(np.max(self.tab[0, 0]))

    def _radial(self, r) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        return self.norm * self._g(r * r) * _cosine_taper(r, self.r_sup)

    def func(self, u, v) -> np.ndarray:
        """The profile at offsets (u, v) from the source (unit integral, support r_sup)."""
        r2 = u * u + v * v
        return self.norm * self._g(r2) * _cosine_taper(np.sqrt(r2), self.r_sup)

    def _blend(self, x: float, y: float):
        p = PROFILE_PHASES
        nx0, ny0 = int(np.floor(x)), int(np.floor(y))
        gx, gy = p * (float(x) - nx0), p * (float(y) - ny0)
        i0, k0 = min(int(np.floor(gx)), p - 1), min(int(np.floor(gy)), p - 1)
        return nx0, ny0, i0, k0, gx - i0, gy - k0

    def placed(self, x: float, y: float):
        """(stamp, nx0, ny0): a source at (x, y) as the bilinear blend of its four nearest phase stamps, centre
        pixel (nx0, ny0) = (floor x, floor y)."""
        nx0, ny0, i0, k0, tx, ty = self._blend(x, y)
        t = self.tab
        st = ((1 - tx) * (1 - ty) * t[k0, i0] + tx * (1 - ty) * t[k0, i0 + 1]
              + (1 - tx) * ty * t[k0 + 1, i0] + tx * ty * t[k0 + 1, i0 + 1])
        return st, nx0, ny0

    def placed_cut(self, x: float, y: float, cx: int, cy: int, half: int):
        """The part of ``placed(x, y)`` inside the (2 half + 1)^2 box centred on pixel (cx, cy):
        (box rows, box cols, values), or None when they do not overlap (only that part is blended)."""
        nx0, ny0, i0, k0, tx, ty = self._blend(x, y)
        m = self.m
        x0, x1 = max(cx - half, nx0 - m), min(cx + half, nx0 + m)
        y0, y1 = max(cy - half, ny0 - m), min(cy + half, ny0 + m)
        if x0 > x1 or y0 > y1:
            return None
        sy = slice(y0 - (ny0 - m), y1 - (ny0 - m) + 1)
        sx = slice(x0 - (nx0 - m), x1 - (nx0 - m) + 1)
        t = self.tab
        st = ((1 - tx) * (1 - ty) * t[k0, i0, sy, sx] + tx * (1 - ty) * t[k0, i0 + 1, sy, sx]
              + (1 - tx) * ty * t[k0 + 1, i0, sy, sx] + tx * ty * t[k0 + 1, i0 + 1, sy, sx])
        return (slice(y0 - (cy - half), y1 - (cy - half) + 1), slice(x0 - (cx - half), x1 - (cx - half) + 1), st)


class MoffatProfile(_TabulatedProfile):
    """The iteration-0 neighbour profile of round 5 (prereg_a.3_r5.md (ga), kept by r5d): Moffat (fwhm, beta),
    support R = alpha sqrt(10^(5 / beta) - 1), cosine taper from 0.8 R to R, unit plane integral, placed by its
    16 x 16 phase table."""

    def __init__(self, fwhm: float, beta: float):
        self.fwhm, self.beta = float(fwhm), float(beta)
        self.alpha = _moffat_alpha(self.fwhm, self.beta)
        self._a2 = self.alpha * self.alpha
        self._k = (self.beta - 1.0) / (np.pi * self._a2)
        self._build(self.alpha * np.sqrt(PROFILE_SUPPORT_DROP ** (-1.0 / self.beta) - 1.0))

    def _g(self, r2):
        return self._k * (1.0 + r2 / self._a2) ** (-self.beta)


class GaussianProfile(_TabulatedProfile):
    """The fallback profile when fewer than EMPIRICAL_STAMP_MIN_STARS isolated training stars give no empirical
    stamp: a circular Gaussian of the FWHM mode, support where it falls to PROFILE_SUPPORT_DROP of its peak,
    otherwise as :class:`MoffatProfile` (not exercised by the P5 scenes; prereg_a.3_r5d.md 7)."""

    def __init__(self, fwhm: float):
        self.fwhm = float(fwhm)
        self._s2 = (self.fwhm / _FWHM_PER_SIGMA) ** 2
        self._build(np.sqrt(2.0 * self._s2 * np.log(1.0 / PROFILE_SUPPORT_DROP)))

    def _g(self, r2):
        return np.exp(-0.5 * r2 / self._s2) / (2.0 * np.pi * self._s2)


def fit_moffat(stamp) -> tuple[float, float]:
    """(fwhm, beta) of the round-5 (ga) fit (prereg_a.3_r5.md 1): least squares of the pixel-integrated
    (Gauss-Legendre 3 x 3) Moffat centred on the stamp's centre pixel to the stamp, amplitude in closed form;
    bounds fwhm 1-20 px, beta 1.05-20.  Start: beta 2.5 and the FWHM of the stamp's half-maximum area
    (2 sqrt(A / pi), A = pixels >= half the peak), clipped to the bounds."""
    from scipy.optimize import least_squares

    st = np.asarray(stamp, dtype=np.float64)
    n = st.shape[0]
    fw0 = float(np.clip(2.0 * np.sqrt(np.count_nonzero(st >= 0.5 * st.max()) / np.pi), 1.5, 19.0))

    def resid(par):
        mod = _pixel_integrate(lambda u, v: _moffat_raw(u, v, par[0], par[1]), 0.0, 0.0, n)
        amp = float(np.sum(mod * st) / np.sum(mod * mod))
        return (amp * mod - st).ravel()

    fw, be = least_squares(resid, x0=[fw0, 2.5], bounds=([1.0, 1.05], [20.0, 20.0])).x
    return float(fw), float(be)


def radial_profile(stamp) -> MoffatProfile:
    """The iteration-0 neighbour profile of an image (prereg_a.3_r5.md (ga), kept by r5d): the Moffat fitted to
    its empirical stamp (:func:`fit_moffat`), tabulated by :class:`MoffatProfile`."""
    return MoffatProfile(*fit_moffat(stamp))


#: Empirical neighbour stamp of iteration 0 (design v2.5 3rd ed. 5.3): the median of at most
#: this many registered stamps of the brightest isolated training stars ...
EMPIRICAL_STAMP_MAX_STARS = 50
#: ... and at least this many, else the FWHM-mode Gaussian (logged).
EMPIRICAL_STAMP_MIN_STARS = 5


def empirical_star_stamp(stamps, flux, isolated, *, n_max: int = EMPIRICAL_STAMP_MAX_STARS,
                         n_min: int = EMPIRICAL_STAMP_MIN_STARS):
    """Pixelwise median (normalised to sum 1) of the registered stamps of the ``n_max`` brightest
    (by ``flux``) training stars with ``isolated`` true; None when fewer than ``n_min`` are."""
    idx = np.flatnonzero(np.asarray(isolated, dtype=bool))
    if len(idx) < int(n_min):
        return None
    flux = np.asarray(flux, dtype=np.float64)
    top = idx[np.argsort(-flux[idx])][:int(n_max)]
    med = np.median(np.asarray(stamps, dtype=np.float64)[top], axis=0)
    return med / med.sum()


def select_psf_stars(
    objs_all: np.ndarray,
    *,
    image_shape: tuple[int, int],
    fwhm_mode: float,
    min_snr: float,
    edge_pad_pix: int,
    cutout_size: int,
    psf_size: int,
    bad_mask: np.ndarray,
    sat_mask: np.ndarray,
    linear_limit: float,
    rows: TrainingRows,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    """The PSF training stars of one image (design 1.5).

    Returns (indices into ``objs_all``, counts after each rule, paired index into ``rows``).
    Rules, in order (the counts are the detections passing every rule up to that one):

    1. ``after_edge``: the star's cutout (``edge_pad_pix`` from the border) is in the image.
    2. ``after_identity``: paired one-to-one with a trainable input row
       (:func:`pair_detections_with_rows`, radius :data:`TRAINING_MATCH_RADIUS_FWHM` x FWHM).
    3. ``after_snr``: SEP S/N >= ``min_snr`` (the candidate pool, D9-4).
    4. ``after_saturation``: no pixel of ``sat_mask`` in the star's cutout (``cutout_size``,
       the pixels its stamp is made from).
    5. ``after_linear_limit``: SEP peak < ``linear_limit`` (image units; inf = no limit).
    6. ``after_bad_pixels``: no pixel of ``bad_mask`` in the cutout.
    7. ``after_brighter_neighbour``: no brighter detection's peak pixel in the star's
       ``psf_size`` box.

    These are properties of the data; cleanliness is a continuous weight
    (:func:`psf0_star_noise`), so there is no ellipticity cut and no separation rule between
    candidates (removed in M2 unit a.3).
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    n = int(len(objs_all))
    counts: dict[str, int] = {"total": n}
    steps = ("after_edge", "after_identity", "after_snr", "after_saturation",
             "after_linear_limit", "after_bad_pixels", "after_brighter_neighbour")
    none = np.zeros(0, dtype=np.int64)
    if n == 0:
        counts.update({k: 0 for k in steps})
        return none, counts, none
    if not (np.isfinite(fwhm_mode) and fwhm_mode > 0):
        raise ValueError(f"select_psf_stars: FWHM mode {fwhm_mode!r} gives no pairing radius")
    x, y, _, _, snr = _objs_with_quality_metrics(objs_all)
    flux = np.asarray(objs_all["flux"], dtype=np.float64)
    ix = np.rint(x).astype(np.int64)
    iy = np.rint(y).astype(np.int64)
    half_cut = int(cutout_size) // 2

    pad = int(edge_pad_pix)
    keep = (x > pad) & (x < W - pad - 1) & (y > pad) & (y < H - pad - 1)
    counts["after_edge"] = int(keep.sum())

    pair = pair_detections_with_rows(x, y, rows.x, rows.y, TRAINING_MATCH_RADIUS_FWHM * float(fwhm_mode))
    trainable = np.asarray(rows.trainable, dtype=bool)
    paired = pair >= 0
    if len(trainable):
        keep &= paired & trainable[np.maximum(pair, 0)]
    else:
        keep &= False
    counts["after_identity"] = int(keep.sum())

    keep &= snr >= float(min_snr)
    counts["after_snr"] = int(keep.sum())

    keep &= ~_any_in_box(np.asarray(sat_mask, dtype=bool), ix, iy, half_cut, keep)
    counts["after_saturation"] = int(keep.sum())

    if np.isfinite(linear_limit):
        if "peak" not in (objs_all.dtype.names or ()):
            raise ValueError("select_psf_stars: a linear limit needs the SEP 'peak' field")
        keep &= np.asarray(objs_all["peak"], dtype=np.float64) < float(linear_limit)
    counts["after_linear_limit"] = int(keep.sum())

    keep &= ~_any_in_box(np.asarray(bad_mask, dtype=bool), ix, iy, half_cut, keep)
    counts["after_bad_pixels"] = int(keep.sum())

    keep &= ~_brighter_peak_in_box(objs_all, flux, ix, iy, int(psf_size) // 2, keep)
    counts["after_brighter_neighbour"] = int(keep.sum())

    sel = np.flatnonzero(keep).astype(np.int64)
    return sel, counts, pair[sel]


def _box_own_noise(sigma, ix: int, iy: int, half: int, s: int) -> float:
    box = np.asarray(sigma[iy - half:iy + half + 1, ix - half:ix + half + 1], dtype=np.float64)
    if box.shape != (2 * half + 1, 2 * half + 1):
        raise ValueError(f"star {s}: its box leaves the image")
    return float(np.mean(box * box))


class _PlacedStamps:
    """Unit stamps of neighbours placed at their positions: a centred (2 half + 1)^2 stamp moved by
    the sub-pixel part of the position with :func:`fourier_shift` and put with its centre on the
    position's nearest pixel (cached per neighbour)."""

    def __init__(self, x, y, stamp_of, half: int):
        self.x, self.y, self.stamp_of, self.half = x, y, stamp_of, int(half)
        self.cache: dict = {}

    def __call__(self, j: int):
        if j not in self.cache:
            ix, iy = int(np.rint(self.x[j])), int(np.rint(self.y[j]))
            st = np.asarray(self.stamp_of(j), dtype=np.float64)
            if st.shape != (2 * self.half + 1, 2 * self.half + 1):
                raise ValueError(f"neighbour stamp of shape {st.shape}; expected the box size")
            self.cache[j] = (fourier_shift(st, self.x[j] - ix, self.y[j] - iy), ix, iy)
        return self.cache[j]


def _overlap(ix_s: int, iy_s: int, ix_j: int, iy_j: int, half: int, half_j: int | None = None):
    """Slices of the star box (centre ix_s, iy_s, half-size ``half``) and of a neighbour stamp
    (centre ix_j, iy_j, half-size ``half_j``, default ``half``) that cover the same pixels, or None."""
    hj = half if half_j is None else int(half_j)
    # box pixel offsets -half..half; stamp pixel offsets -hj..hj around (ix_j, iy_j)
    x0 = max(-half, ix_j - ix_s - hj)
    x1 = min(half, ix_j - ix_s + hj)
    y0 = max(-half, iy_j - iy_s - hj)
    y1 = min(half, iy_j - iy_s + hj)
    if x0 > x1 or y0 > y1:
        return None
    bx = slice(x0 + half, x1 + half + 1)
    by = slice(y0 + half, y1 + half + 1)
    sx = slice(x0 - (ix_j - ix_s) + hj, x1 - (ix_j - ix_s) + hj + 1)
    sy = slice(y0 - (iy_j - iy_s) + hj, y1 - (iy_j - iy_s) + hj + 1)
    return (by, bx), (sy, sx)


def psf0_star_noise(star_x, star_y, star_flux, star_det, det_x, det_y, det_flux, sigma,
                    fwhm_mode: float, *, half: int = 27, profile: _TabulatedProfile | None = None):
    """Noise of the normalised training stamps of PSF^(0) (design 1.5 v2.4 D6; v2.5 a_s; 3rd ed. 5.3).

    For star s (position star_x/y, normalising flux F_s = the box sum its stamp was divided by,
    detection index star_det[s] among the SEP detections det_*):
      sigma2_own = mean over the (2 half + 1)^2 box centred on (rint x_s, rint y_s) of sigma^2;
      sigma2_nb  = mean over the box of (sum_{j != star_det[s]} F_j G_j(p))^2 -- iteration 0
                   subtracts no neighbour, so its contamination is the whole neighbour light (D6);
                   F_j = det_flux (in round 5d the fitted amplitudes of the kept neighbours: this is
                   the energy step of prereg_a.3_r5d.md 1.4);
                   G_j = ``profile`` placed by its phase table (:meth:`MoffatProfile.placed`, the
                   round-5 (ga) Moffat of :func:`radial_profile`), or when ``profile`` is None the
                   unit-integral circular Gaussian of FWHM ``fwhm_mode`` at pixel centres; both
                   continuous in r_j;
      a_s        = n_pix (sigma2_own + sigma2_nb) / F_s^2, n_pix = (2 half + 1)^2.
    The star weight is u_s/(a_s + tau^2) inside :func:`tract7dt.fit.grid_gpu.psf_model.fit_mixture`.
    Returns (a, sigma2_own, sigma2_nb), each (n,).
    """
    from scipy.spatial import cKDTree

    star_x = np.asarray(star_x, dtype=np.float64)
    star_y = np.asarray(star_y, dtype=np.float64)
    star_flux = np.asarray(star_flux, dtype=np.float64)
    star_det = np.asarray(star_det, dtype=np.int64)
    det_x = np.asarray(det_x, dtype=np.float64)
    det_y = np.asarray(det_y, dtype=np.float64)
    det_flux = np.asarray(det_flux, dtype=np.float64)
    sigma = np.asarray(sigma)
    half = int(half)
    n = len(star_x)
    n_pix = (2 * half + 1) ** 2
    own = np.empty(n)
    nb = np.zeros(n)
    tree = cKDTree(np.column_stack([det_x, det_y])) if len(det_x) else None
    off = np.arange(-half, half + 1, dtype=np.float64)
    if profile is not None:
        m_prof = int(profile.m)                      # half-size of the placed profile stamps
        reach = half + m_prof + 1.0
    else:
        s2 = (float(fwhm_mode) / _FWHM_PER_SIGMA) ** 2
        norm = 1.0 / (2.0 * np.pi * s2)
        reach = half + GAUSS_REACH_SIGMA * np.sqrt(s2)
    for s in range(n):
        ix, iy = int(np.rint(star_x[s])), int(np.rint(star_y[s]))
        own[s] = _box_own_noise(sigma, ix, iy, half, s)
        if tree is None:
            continue
        js = [j for j in tree.query_ball_point([float(ix), float(iy)], r=reach, p=np.inf)
              if j != star_det[s]]
        if not js:
            continue
        if profile is not None:
            light = np.zeros((2 * half + 1, 2 * half + 1))
            for j in sorted(js):
                st, jx, jy = profile.placed(det_x[j], det_y[j])
                ov = _overlap(ix, iy, jx, jy, half, m_prof)
                if ov is not None:
                    light[ov[0]] += det_flux[j] * st[ov[1]]
        else:
            js = np.asarray(js, dtype=np.int64)
            gx = np.exp(-0.5 * (ix + off[None, :] - det_x[js, None]) ** 2 / s2)    # (n_j, nx)
            gy = np.exp(-0.5 * (iy + off[None, :] - det_y[js, None]) ** 2 / s2)    # (n_j, ny)
            light = (gy.T * (det_flux[js] * norm)) @ gx                           # (ny, nx)
        nb[s] = float(np.mean(light * light))
    a = n_pix * (own + nb) / star_flux ** 2
    return a, own, nb


def neighbour_amplitude(sigma_flux, flux, s_b=0.0) -> np.ndarray:
    """Amplitude c_j of a subtracted neighbour's light for iterations t >= 1 (design 1.5, D6;
    3rd ed. 5.2): c_j^2 = sigma_Fj^(t-1)^2 + (s_b(r_j)^(t-1) F_j^(t-1))^2, s_b^(-1) = 0
    (``s_b`` a number or one value per neighbour)."""
    return np.hypot(np.asarray(sigma_flux, dtype=np.float64),
                    np.asarray(s_b, dtype=np.float64) * np.asarray(flux, dtype=np.float64))


def star_noise_t(star_x, star_y, star_flux, star_self, nb_x, nb_y, nb_sigma_flux, nb_flux, sigma,
                 psf_at, *, nb_s_b=None, half: int = 27):
    """Noise of the normalised training stamps of PSF^(t), t >= 1 (design 1.5 D6; 3rd ed. 5.2).

    The stamps have every neighbour subtracted, so what is left is the subtraction error, and the
    errors of different neighbours are independent:
      sigma2_nb,s = (1/n_pix) sum_{p in box} sum_{j != s} c_j^2 P_j(p)^2,
      c_j^2 = sigma_Fj^2 + (s_b(r_j) F_j)^2 (:func:`neighbour_amplitude`; s_b None = 0),
    P_j = ``psf_at(x_j, y_j)`` -- the unit-sum centred stamp of PSF^(t-1) at the neighbour --
    moved to r_j as in :func:`psf0_star_noise`.  star_self[s] = the star's own index among the
    neighbours (-1: not among them).  sigma2_own and a_s as in :func:`psf0_star_noise`.
    Returns (a, sigma2_own, sigma2_nb).
    """
    from scipy.spatial import cKDTree

    star_x = np.asarray(star_x, dtype=np.float64)
    star_y = np.asarray(star_y, dtype=np.float64)
    star_flux = np.asarray(star_flux, dtype=np.float64)
    star_self = np.asarray(star_self, dtype=np.int64)
    nb_x = np.asarray(nb_x, dtype=np.float64)
    nb_y = np.asarray(nb_y, dtype=np.float64)
    s_b = 0.0 if nb_s_b is None else np.asarray(nb_s_b, dtype=np.float64)
    c2 = neighbour_amplitude(nb_sigma_flux, nb_flux, s_b) ** 2
    sigma = np.asarray(sigma)
    half = int(half)
    n = len(star_x)
    n_pix = (2 * half + 1) ** 2
    own = np.empty(n)
    nb = np.zeros(n)
    placed = _PlacedStamps(nb_x, nb_y, lambda j: psf_at(nb_x[j], nb_y[j]), half)
    tree = cKDTree(np.column_stack([nb_x, nb_y])) if len(nb_x) else None
    for s in range(n):
        ix, iy = int(np.rint(star_x[s])), int(np.rint(star_y[s]))
        own[s] = _box_own_noise(sigma, ix, iy, half, s)
        if tree is None:
            continue
        acc = np.zeros((2 * half + 1, 2 * half + 1))
        for j in sorted(tree.query_ball_point([float(ix), float(iy)], r=2 * half + 1.0, p=np.inf)):
            if j == star_self[s]:
                continue
            st, jx, jy = placed(j)
            ov = _overlap(ix, iy, jx, jy, half)
            if ov is not None:
                acc[ov[0]] += c2[j] * st[ov[1]] ** 2
        nb[s] = float(np.mean(acc))
    a = n_pix * (own + nb) / star_flux ** 2
    return a, own, nb


def star_noise_a(box_sum, sky_var, sigma2_nb, *, kappa: float, phi_norm2, gain: float,
                 n_pix: int = 3025) -> np.ndarray:
    """a_s, the expected squared norm of the noise of a stamp normalised by its box sum S_s (design
    v2.5 3rd ed. 7.1):
      a_s = n_pix sigma2_sky,s (1 + kappa |phi|^2) / S^2 + 1 / (g S) + n_pix sigma2_nb,s / S^2.
    sky_var: the box-mean sky variance; kappa |phi|^2: the share of the box-sum noise that the
    normalisation puts into the shape (kappa = empty-box sum variance / (n_pix sigma^2),
    :func:`box_sum_kappa`; phi = the normalised PSF estimate: the empirical stamp at t = 0, the
    previous model at t >= 1); 1 / (g S): the star's own Poisson noise (g = gain, electrons per
    image unit); sigma2_nb: the neighbour term (:func:`psf0_star_noise` at t = 0,
    :func:`star_noise_t` at t >= 1).  The tests' fixtures use this same function."""
    box_sum = np.asarray(box_sum, dtype=np.float64)
    sky = np.asarray(sky_var, dtype=np.float64)
    nb = np.asarray(sigma2_nb, dtype=np.float64)
    phi2 = np.asarray(phi_norm2, dtype=np.float64)
    return (n_pix * sky * (1.0 + float(kappa) * phi2) / box_sum ** 2 + 1.0 / (float(gain) * box_sum)
            + n_pix * nb / box_sum ** 2)


#: Fewer empty boxes than this: kappa = 1 (no box-sum correlation), logged by the caller.
KAPPA_MIN_BOXES = 20


def box_sum_kappa(image, sigma_sky, objs, bad_mask, *, half: int = 27):
    """kappa = variance of empty-box sums / (n_pix sigma_sky^2) (design v2.5 3rd ed. 7.1).

    The image is tiled by (2 half + 1)^2 boxes from its corner; a box is empty when no detection's
    peak pixel and no bad pixel lies in it.  Variance = (1.4826 MAD)^2 of the box sums (robust to
    faint sources below the detection threshold); sigma_sky^2 = the median of the boxes' mean sky
    variance.  Returns (kappa, n_empty_boxes); fewer than :data:`KAPPA_MIN_BOXES`: (1.0, n)."""
    img = np.asarray(image, dtype=np.float64)
    sky = np.asarray(sigma_sky, dtype=np.float64)
    bad = np.asarray(bad_mask, dtype=bool)
    n = 2 * int(half) + 1
    hh, ww = img.shape
    ny, nx = hh // n, ww // n
    occupied = np.zeros((ny, nx), dtype=bool)
    names = objs.dtype.names or ()
    if len(objs):
        px = np.asarray(objs["xpeak"] if "xpeak" in names else np.rint(objs["x"]), dtype=np.int64)
        py = np.asarray(objs["ypeak"] if "ypeak" in names else np.rint(objs["y"]), dtype=np.int64)
        ok = (px >= 0) & (py >= 0) & (px < nx * n) & (py < ny * n)
        occupied[py[ok] // n, px[ok] // n] = True
    blocks = lambda a: a[:ny * n, :nx * n].reshape(ny, n, nx, n)
    occupied |= blocks(bad).any(axis=(1, 3))
    sums = blocks(img).sum(axis=(1, 3))[~occupied]
    if sums.size < KAPPA_MIN_BOXES:
        return 1.0, int(sums.size)
    var_sum = (1.4826 * float(np.median(np.abs(sums - np.median(sums))))) ** 2
    sky_var = float(np.median((blocks(sky) ** 2).mean(axis=(1, 3))[~occupied]))
    return var_sum / (n * n * sky_var), int(sums.size)


# ---------------------------------------------------------------------------
# Iteration-0 neighbour term (M2 unit a.3 round 5d, prereg_a.3_r5d.md 1.1-1.3 and its addendum A.3):
# a well-posed joint linear fit of the KNOWN sources in every training-star box.
# ---------------------------------------------------------------------------

#: List rows closer than this many Moffat FWHM (Euclidean, <=; connected components) are one source (r5d 1.1).
R_DUP_FWHM = 0.10
#: A SEP detection with no list row within this many FWHM-mode (Euclidean, <= counts as listed) is an unlisted
#: known source at its centroid (r5c / r5d 1.1).
R_U_FWHM = 1.0
#: Central-difference step (px) of the own source's two shift columns (r5d 1.2).
SHIFT_H = 0.25
#: The own position is updated by Gauss-Newton steps projected on the disc of this radius (px) around the
#: measured centre ...
RECENTRE_MAX = 1.0
#: ... stopping at the first x_k (k >= 1) whose next projected step is <= this (px) ...
RECENTRE_TOL = 0.01
#: ... after at most this many updates (r5d addendum A.3).
RECENTRE_ITERS = 5
#: Plausible-amplitude scale F_u = F_MARGIN (max(0, 3 x 3 max) + F_NSIG sigma) / phi_max (r5d 1.2).
F_MARGIN = 10.0
F_NSIG = 5.0
#: A neighbour column is dropped when F_u ||c_u||_w < T_DROP (its ridge would shrink it by > 1 - 1e-4) (r5d 1.2).
T_DROP = 0.01
#: A box whose equilibrated, ridge-augmented system has a larger condition number is degenerate (r5d 1.2).
COND_DEGENERATE = 1e10
#: The fast solve (FP64 Gram + Cholesky + two refinement steps on the full system) is used when LAPACK's
#: 1-norm estimate of the equilibrated Gram's reciprocal condition is at least this (cond(system) <~ 1e5,
#: where two FP64 refinement steps leave ~cond^2 eps^2 relative); other boxes take the reference solve
#: (SVD lstsq of the equilibrated, ridge-augmented system), which also decides degeneracy exactly.
FAST_SOLVE_MIN_RCOND = 1e-10
#: Distance bins (in Moffat FWHM from the measured centre) of the run.log diagnostic of r5d A.2 / A.11.
NEIGHBOUR_BINS_FWHM = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0))


@dataclass(frozen=True)
class KnownSources:
    """The known sources U of iteration 0 (prereg_a.3_r5d.md 1.1): list-row components first (ordered by their
    smallest row index, each at the unweighted mean of its rows), then the unlisted detections in detection order
    (at their centroids).  comp_of_row[k] = the source of list row k; det_src[d] = the source of detection d (the
    component of its nearest row when that row is within r_u, ties to the lower row index, else the detection)."""
    x: np.ndarray
    y: np.ndarray
    comp_of_row: np.ndarray
    det_src: np.ndarray
    unlisted: np.ndarray
    n_comp: int
    n_rows_merged: int
    r_dup: float
    r_u: float
    det_row_dist: np.ndarray


def _dup_components(x, y, r_dup: float) -> np.ndarray:
    """Smallest row index of each row's connected component under 'Euclidean distance <= r_dup'."""
    from scipy.spatial import cKDTree

    n = len(x)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n > 1 and r_dup > 0:
        pairs = cKDTree(np.column_stack([x, y])).query_pairs(r=float(r_dup), output_type="ndarray")
        for a, b in pairs:
            # exact distance (the tree's own test is not guaranteed bit-exact at the boundary)
            if (x[a] - x[b]) ** 2 + (y[a] - y[b]) ** 2 <= r_dup * r_dup:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def known_sources(row_x, row_y, det_x, det_y, *, fwhm: float, fwhm_mode: float) -> KnownSources:
    """The known sources of iteration 0 (prereg_a.3_r5d.md 1.1): list rows (any TYPE, blank rows included, no list
    flux used) merged within R_DUP_FWHM x ``fwhm`` (the Moffat FWHM), plus every detection with no row within
    R_U_FWHM x ``fwhm_mode``."""
    from scipy.spatial import cKDTree

    rx = np.asarray(row_x, dtype=np.float64)
    ry = np.asarray(row_y, dtype=np.float64)
    dx = np.asarray(det_x, dtype=np.float64)
    dy = np.asarray(det_y, dtype=np.float64)
    r_dup = R_DUP_FWHM * float(fwhm)
    root = _dup_components(rx, ry, r_dup)
    reps = np.unique(root)
    comp_of_row = np.searchsorted(reps, root)
    cnt = np.bincount(comp_of_row, minlength=len(reps)).astype(np.float64)
    cx = np.bincount(comp_of_row, weights=rx, minlength=len(reps)) / np.maximum(cnt, 1)
    cy = np.bincount(comp_of_row, weights=ry, minlength=len(reps)) / np.maximum(cnt, 1)
    r_u = R_U_FWHM * float(fwhm_mode)
    nd = len(dx)
    dist = np.full(nd, np.inf)
    nearest = np.full(nd, -1, dtype=np.int64)
    if len(rx) and nd:
        tree = cKDTree(np.column_stack([rx, ry]))
        _, cand = tree.query(np.column_stack([dx, dy]), k=1)
        for d in range(nd):
            # exact distances to every row as close as the tree's nearest (ties -> the lower row index)
            r0 = float(np.hypot(rx[cand[d]] - dx[d], ry[cand[d]] - dy[d]))
            near = np.asarray(tree.query_ball_point([dx[d], dy[d]], r=r0 * (1.0 + 1e-12) + 1e-12), dtype=np.int64)
            dd = np.hypot(rx[near] - dx[d], ry[near] - dy[d])
            j = np.lexsort((near, dd))[0]
            nearest[d], dist[d] = near[j], dd[j]
    listed = dist <= r_u
    unl = np.flatnonzero(~listed)
    det_src = np.full(nd, -1, dtype=np.int64)
    det_src[listed] = comp_of_row[nearest[listed]]
    det_src[unl] = len(reps) + np.arange(len(unl))
    return KnownSources(x=np.concatenate([cx, dx[unl]]), y=np.concatenate([cy, dy[unl]]), comp_of_row=comp_of_row,
                        det_src=det_src, unlisted=unl, n_comp=int(len(reps)),
                        n_rows_merged=int(np.sum(cnt[comp_of_row] > 1)), r_dup=r_dup, r_u=r_u, det_row_dist=dist)


def plausible_amplitude(image, sky_sigma, zero_w, x, y, phi_max: float) -> np.ndarray:
    """F_u (prereg_a.3_r5d.md 1.2): F_MARGIN x (max(0, max of the data on the 3 x 3 pixels around
    (rint x_u, rint y_u)) + F_NSIG x the largest sky sigma there) / phi_max; +inf (no ridge, never dropped) when one
    of the nine pixels lies outside the image or has zero weight.  Measured on the image, never the list flux."""
    h, w = image.shape
    ix = np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)
    iy = np.rint(np.asarray(y, dtype=np.float64)).astype(np.int64)
    F = np.full(len(ix), np.inf)
    ok = np.flatnonzero((ix >= 1) & (ix <= w - 2) & (iy >= 1) & (iy <= h - 2))
    if ok.size:
        oy, ox = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
        yy = iy[ok][:, None] + oy.ravel()[None, :]
        xx = ix[ok][:, None] + ox.ravel()[None, :]
        dv = np.asarray(image[yy, xx], dtype=np.float64)
        sv = np.asarray(sky_sigma[yy, xx], dtype=np.float64)
        zw = np.asarray(zero_w[yy, xx], dtype=bool)
        val = F_MARGIN * (np.maximum(0.0, dv.max(axis=1)) + F_NSIG * sv.max(axis=1)) / float(phi_max)
        F[ok] = np.where(zw.any(axis=1), np.inf, val)
    return F


def _solve_box(Xo, XE, dw, GE, rE, ridgeE):
    """The unique minimiser of |dw - [Xo XE^T] theta|^2 + sum ridgeE theta_E^2 (weighted rows; columns ordered own
    (3 rows of Xo), then the rows of XE = the kept neighbours and the constant last; GE = XE XE^T, rE = XE dw).
    Returns (theta, cond, degenerate)."""
    import scipy.linalg as sla

    nc = 3 + XE.shape[0]
    G = np.empty((nc, nc))
    G[:3, :3] = Xo @ Xo.T
    G[:3, 3:] = Xo @ XE.T
    G[3:, :3] = G[:3, 3:].T
    G[3:, 3:] = GE
    rhs = np.concatenate([Xo @ dw, rE])
    ridge = np.concatenate([np.zeros(3), ridgeE])
    A = G
    A[np.diag_indices(nc)] += ridge
    dg = np.diag(A).copy()
    if np.all(dg > 0):
        s = 1.0 / np.sqrt(dg)
        Ae = A * s[:, None] * s[None, :]
        try:
            c, low = sla.cho_factor(Ae, lower=True, check_finite=False)
            rcond, info = sla.lapack.dpocon(c, float(np.max(np.sum(np.abs(Ae), axis=0))), uplo="L")
        except np.linalg.LinAlgError:
            rcond, info = 0.0, 1
        if info == 0 and rcond >= FAST_SOLVE_MIN_RCOND:
            theta = s * sla.cho_solve((c, low), s * rhs, check_finite=False)
            for _ in range(2):           # refinement on the full weighted system (not the normal equations)
                res = dw - (theta[:3] @ Xo + theta[3:] @ XE)
                g = np.concatenate([Xo @ res, XE @ res]) - ridge * theta
                theta = theta + s * sla.cho_solve((c, low), s * g, check_finite=False)
            return theta, float(np.sqrt(1.0 / rcond)), False
    # reference solve: equilibrated, ridge-augmented system by SVD lstsq (exact condition number, minimum norm)
    X = np.concatenate([Xo, XE]).T
    scale = 1.0 / np.sqrt(np.sum(X * X, axis=0) + ridge)
    rr = np.flatnonzero(ridge > 0)
    aug = np.concatenate([X * scale[None, :], (np.sqrt(ridge[rr]) * scale[rr])[:, None] * np.eye(nc)[rr]])
    sol, _, rank, sv = np.linalg.lstsq(aug, np.concatenate([dw, np.zeros(rr.size)]), rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    return sol * scale, cond, bool(cond > COND_DEGENERATE or rank < nc)


def fit_iteration0_boxes(image, sky_sigma, zero_w, box_x, box_y, start_x, start_y, own_src, src_x, src_y, F,
                         profile: _TabulatedProfile, *, half: int = 27, return_fits: bool = False) -> dict:
    """The round-5d joint fit (prereg_a.3_r5d.md 1.2 + addendum A.3) on the (2 half + 1)^2 box centred on each
    (box_x, box_y).

    Columns: the own source ``own_src`` placed at its position x_k with two shift columns
    (c(x + h) - c(x - h)) / 2h (h = SHIFT_H); every other known source within Chebyshev half + m + 1 of the box
    centre whose weighted norm ||c_u||_w is > 0 and with F_u ||c_u||_w >= T_DROP (F_u = +inf: kept, no ridge); one
    constant.  Weight 1 / sky_sigma^2, 0 on ``zero_w`` and outside the image.  Minimiser of the weighted chi^2 +
    sum_u (A_u / F_u)^2.  Own position: x_0 = (start_x, start_y); x_{k+1} = x_k + (B_x, B_y) / A_o projected on the
    disc of radius RECENTRE_MAX around x_0; the returned fit is the solve at the first x_k (k >= 1) whose next step
    is <= RECENTRE_TOL, else at x_K (K = RECENTRE_ITERS), or at x_k when A_o <= 0 there.
    Returns per box: sigma2_nb (box mean of the fitted neighbour light squared), own (A_o), own_x, own_y, const,
    n_cand, n_kept, cond (1-norm estimate on the fast path, SVD value otherwise), degenerate, n_updates,
    projected, at_edge (projected and the final position on the disc's edge), converged, own_nonpositive; with
    ``return_fits`` also fits[b] = {src, amp} (kept neighbours and their amplitudes)."""
    from scipy.spatial import cKDTree

    half = int(half)
    n = 2 * half + 1
    h, w = image.shape
    m = int(profile.m)
    reach = half + m + 1.0
    src_x = np.asarray(src_x, dtype=np.float64)
    src_y = np.asarray(src_y, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    tree = cKDTree(np.column_stack([src_x, src_y])) if len(src_x) else None
    # per-source phase-table placement (the arithmetic of _TabulatedProfile._blend, vectorised once)
    p = PROFILE_PHASES
    s_nx0 = np.floor(src_x).astype(np.int64)
    s_ny0 = np.floor(src_y).astype(np.int64)
    s_gx = p * (src_x - s_nx0)
    s_gy = p * (src_y - s_ny0)
    s_i0 = np.minimum(np.floor(s_gx).astype(np.int64), p - 1)
    s_k0 = np.minimum(np.floor(s_gy).astype(np.int64), p - 1)
    s_tx = s_gx - s_i0
    s_ty = s_gy - s_k0
    tab = profile.tab
    nb = len(box_x)
    out = {k: np.zeros(nb) for k in ("sigma2_nb", "own", "own_x", "own_y", "const", "cond")}
    for k in ("n_cand", "n_kept", "n_updates"):
        out[k] = np.zeros(nb, dtype=np.int64)
    for k in ("degenerate", "projected", "at_edge", "converged", "own_nonpositive"):
        out[k] = np.zeros(nb, dtype=bool)
    fits = []
    for b in range(nb):
        cx, cy = int(box_x[b]), int(box_y[b])
        ys0, xs0 = cy - half, cx - half
        yy0, yy1, xx0, xx1 = max(0, ys0), min(h, ys0 + n), max(0, xs0), min(w, xs0 + n)
        d = np.zeros((n, n))
        wt = np.zeros((n, n))
        if yy0 < yy1 and xx0 < xx1:
            d[yy0 - ys0:yy1 - ys0, xx0 - xs0:xx1 - xs0] = image[yy0:yy1, xx0:xx1]
            sg = np.asarray(sky_sigma[yy0:yy1, xx0:xx1], dtype=np.float64)
            wt[yy0 - ys0:yy1 - ys0, xx0 - xs0:xx1 - xs0] = np.where(
                np.asarray(zero_w[yy0:yy1, xx0:xx1], dtype=bool), 0.0, 1.0 / (sg * sg))
        own = int(own_src[b])
        cand = [] if tree is None else sorted(tree.query_ball_point([float(cx), float(cy)], r=reach, p=np.inf))
        cand = [u for u in cand if u != own]
        # every candidate's column on the box (only its overlap is blended), then the norm and drop rules at once
        C = np.zeros((len(cand), n, n))
        for j, u in enumerate(cand):
            nx0, ny0 = int(s_nx0[u]), int(s_ny0[u])
            x0, x1 = max(xs0, nx0 - m), min(cx + half, nx0 + m)
            y0, y1 = max(ys0, ny0 - m), min(cy + half, ny0 + m)
            if x0 > x1 or y0 > y1:
                continue
            sy = slice(y0 - (ny0 - m), y1 - (ny0 - m) + 1)
            sx = slice(x0 - (nx0 - m), x1 - (nx0 - m) + 1)
            i0, k0, tx, ty = int(s_i0[u]), int(s_k0[u]), float(s_tx[u]), float(s_ty[u])
            C[j, y0 - ys0:y1 - ys0 + 1, x0 - xs0:x1 - xs0 + 1] = (
                (1 - tx) * (1 - ty) * tab[k0, i0, sy, sx] + tx * (1 - ty) * tab[k0, i0 + 1, sy, sx]
                + (1 - tx) * ty * tab[k0 + 1, i0, sy, sx] + tx * ty * tab[k0 + 1, i0 + 1, sy, sx])
        C = C.reshape(len(cand), n * n)
        cand_arr = np.asarray(cand, dtype=np.int64)
        nrm = np.sqrt((C * C) @ wt.ravel()) if len(cand) else np.zeros(0)
        Fc = F[cand_arr]
        fin = np.isfinite(Fc)
        keep = nrm > 0.0
        keep[fin] &= Fc[fin] * nrm[fin] >= T_DROP
        keep_src = cand_arr[keep]
        Craw = C[keep]
        ridge = np.where(np.isfinite(Fc[keep]), 1.0 / Fc[keep] ** 2, 0.0)
        k = int(keep_src.size)
        sw = np.sqrt(wt.ravel())
        XE = np.empty((k + 1, n * n))         # kept neighbours (weighted), then the constant
        np.multiply(Craw, sw[None, :], out=XE[:k])
        XE[k] = sw
        dw = d.ravel() * sw
        GE = XE @ XE.T
        rE = XE @ dw
        ridgeE = np.concatenate([np.asarray(ridge, dtype=np.float64), [0.0]])

        def own_cols(ox, oy):
            def col(px, py):
                c = np.zeros((n, n))
                cut = profile.placed_cut(px, py, cx, cy, half)
                if cut is not None:
                    c[cut[0], cut[1]] = cut[2]
                return c.ravel()
            c0 = col(ox, oy)
            gx = (col(ox + SHIFT_H, oy) - col(ox - SHIFT_H, oy)) / (2 * SHIFT_H)
            gy = (col(ox, oy + SHIFT_H) - col(ox, oy - SHIFT_H)) / (2 * SHIFT_H)
            return np.stack([c0, gx, gy]) * sw[None, :]

        x0 = np.array([float(start_x[b]), float(start_y[b])])
        x = x0.copy()
        theta, cond, degen = _solve_box(own_cols(*x), XE, dw, GE, rE, ridgeE)
        n_upd, projected, converged = 0, False, False
        while theta[0] > 0:
            xn = x + np.array([theta[1], theta[2]]) / theta[0]
            off = xn - x0
            r_off = float(np.hypot(off[0], off[1]))
            if r_off > RECENTRE_MAX:
                xn = x0 + off * (RECENTRE_MAX / r_off)
                projected = True
            if n_upd >= 1 and float(np.hypot(xn[0] - x[0], xn[1] - x[1])) <= RECENTRE_TOL:
                converged = True
                break
            if n_upd >= RECENTRE_ITERS:
                break
            x = xn
            n_upd += 1
            theta, cond, degen = _solve_box(own_cols(*x), XE, dw, GE, rE, ridgeE)
        amp = theta[3:3 + k]
        if k:
            light = amp @ Craw
            out["sigma2_nb"][b] = float(np.mean(light * light))
        out["own"][b], out["const"][b], out["cond"][b] = theta[0], theta[-1], cond
        out["own_x"][b], out["own_y"][b] = x[0], x[1]
        out["n_cand"][b], out["n_kept"][b], out["n_updates"][b] = len(cand), k, n_upd
        out["degenerate"][b], out["projected"][b], out["converged"][b] = degen, projected, converged
        out["at_edge"][b] = bool(projected and np.hypot(*(x - x0)) > RECENTRE_MAX * (1.0 - 1e-9))
        out["own_nonpositive"][b] = bool(theta[0] <= 0)
        if return_fits:
            fits.append(dict(src=np.asarray(keep_src, dtype=np.int64), amp=amp.copy()))
    if return_fits:
        out["fits"] = fits
    return out


def iteration0_neighbour_bins(star_x, star_y, own_src, src_x, src_y, fwhm: float, inv_a) -> list:
    """r5d A.2 / A.11 diagnostic: for each distance bin of NEIGHBOUR_BINS_FWHM (in ``fwhm`` from the measured
    centre, [lo, hi)), the number of training stars with another known source (not their own) in that bin and those
    stars' share of sum(1/a)."""
    from scipy.spatial import cKDTree

    star_x = np.asarray(star_x, dtype=np.float64)
    star_y = np.asarray(star_y, dtype=np.float64)
    inv_a = np.asarray(inv_a, dtype=np.float64)
    tot = float(np.sum(inv_a))
    hit = np.zeros((len(NEIGHBOUR_BINS_FWHM), len(star_x)), dtype=bool)
    if len(src_x) and len(star_x):
        tree = cKDTree(np.column_stack([src_x, src_y]))
        r_max = max(hi for _, hi in NEIGHBOUR_BINS_FWHM) * float(fwhm)
        for s, near in enumerate(tree.query_ball_point(np.column_stack([star_x, star_y]), r=r_max)):
            for u in near:
                if u == int(own_src[s]):
                    continue
                dist = float(np.hypot(src_x[u] - star_x[s], src_y[u] - star_y[s])) / float(fwhm)
                for i, (lo, hi) in enumerate(NEIGHBOUR_BINS_FWHM):
                    if lo <= dist < hi:
                        hit[i, s] = True
    return [dict(lo=lo, hi=hi, n_stars=int(hit[i].sum()),
                 inv_a_share=float(np.sum(inv_a[hit[i]]) / tot) if tot > 0 else 0.0)
            for i, (lo, hi) in enumerate(NEIGHBOUR_BINS_FWHM)]


def psf0_training_noise(image, objs, star_det, star_x, star_y, box_sum, stamps, sigma_sky, gain: float,
                        bad_mask, fwhm_mode: float, *, rows_x, rows_y, star_row, sat_mask=None, half: int = 27,
                        detection_fits: bool = False):
    """a_s of the PSF^(0) training stars (prereg_a.3_r5d.md 1.1-1.4 with its addendum; a by design v2.5 3rd ed. 7.1).

    The empirical stamp E-bar of the isolated training stars (:func:`isolated_in_box`, :func:`empirical_star_stamp`,
    ranked by :func:`sep_auto_flux`) and the Moffat fitted to it (:func:`radial_profile`); the known sources
    (:func:`known_sources`: input rows ``rows_x``/``rows_y`` merged within R_DUP_FWHM x the Moffat FWHM, plus the
    unlisted detections); their plausible amplitudes (:func:`plausible_amplitude`); in every training star's box
    the joint fit of :func:`fit_iteration0_boxes` (own source = the component of its paired row ``star_row``,
    started at the star's measured centre (star_x, star_y)); sigma2_nb = box mean of the fitted neighbour light
    squared; kappa from empty boxes (:func:`box_sum_kappa`); a by :func:`star_noise_a` with phi = E-bar.  Zero
    weight on ``bad_mask`` | ``sat_mask``.  Fewer than 5 isolated stars: a Gaussian of the FWHM mode for the columns
    and phi (logged).  ``detection_fits``: also fit every detection's box (own source = its known source, started
    at its centroid) for info['neighbour_flux'] (= the own amplitude; tests and diagnostics; sigma2_nb and a are
    the same code with and without it).  ``box_sum`` = the box sums the stamps were normalised by; ``gain`` =
    electrons per image unit.  Returns (a, info)."""
    t0 = time.perf_counter()
    star_det = np.asarray(star_det, dtype=np.int64)
    star_x = np.asarray(star_x, dtype=np.float64)
    star_y = np.asarray(star_y, dtype=np.float64)
    star_row = np.asarray(star_row, dtype=np.int64)
    rows_x = np.asarray(rows_x, dtype=np.float64)
    rows_y = np.asarray(rows_y, dtype=np.float64)
    det_x = np.asarray(objs["x"], dtype=np.float64)
    det_y = np.asarray(objs["y"], dtype=np.float64)
    auto = sep_auto_flux(image, objs, bad_mask)            # ranks the isolated stars of E-bar only
    iso = isolated_in_box(objs, star_det, half)
    emp = empirical_star_stamp(stamps, auto[star_det], iso)
    if emp is not None:
        prof = radial_profile(emp)
        phi2 = float(np.sum(np.asarray(emp, dtype=np.float64) ** 2))
    else:
        logger.warning("iteration 0: %d isolated training stars (< %d): Gaussian of the FWHM mode %.3f px for the "
                       "neighbour columns and phi", int(iso.sum()), EMPIRICAL_STAMP_MIN_STARS, float(fwhm_mode))
        prof = GaussianProfile(float(fwhm_mode))
        s2 = (float(fwhm_mode) / _FWHM_PER_SIGMA) ** 2
        off = np.arange(-half, half + 1, dtype=np.float64)
        g = np.exp(-0.5 * (off[:, None] ** 2 + off[None, :] ** 2) / s2) / (2.0 * np.pi * s2)
        phi2 = float(np.sum((g / g.sum()) ** 2))
    t_prof = time.perf_counter()
    zero_w = np.asarray(bad_mask, dtype=bool)
    if sat_mask is not None:
        zero_w = zero_w | np.asarray(sat_mask, dtype=bool)
    U = known_sources(rows_x, rows_y, det_x, det_y, fwhm=prof.fwhm, fwhm_mode=fwhm_mode)
    F = plausible_amplitude(image, sigma_sky, zero_w, U.x, U.y, prof.peak)
    own_src = U.comp_of_row[star_row] if len(star_row) else np.zeros(0, dtype=np.int64)
    fs = fit_iteration0_boxes(image, sigma_sky, zero_w, np.rint(star_x).astype(np.int64),
                              np.rint(star_y).astype(np.int64), star_x, star_y, own_src, U.x, U.y, F, prof, half=half)
    t_fit = time.perf_counter()
    own = np.array([_box_own_noise(sigma_sky, int(np.rint(star_x[s])), int(np.rint(star_y[s])), half, s)
                    for s in range(len(star_x))])
    nb = fs["sigma2_nb"]
    kappa, n_boxes = box_sum_kappa(image, sigma_sky, objs, bad_mask, half=half)
    a = star_noise_a(box_sum, own, nb, kappa=kappa, phi_norm2=phi2, gain=gain, n_pix=(2 * half + 1) ** 2)
    info = {"profile": "moffat" if emp is not None else "gaussian", "n_isolated": int(iso.sum()),
            "moffat_fwhm": float(prof.fwhm), "moffat_beta": float(getattr(prof, "beta", np.nan)),
            "sigma2_sky": own, "sigma2_nb": nb, "kappa": kappa, "kappa_boxes": n_boxes, "phi_norm2": phi2,
            "auto_flux": auto, "n_known": int(len(U.x)), "n_unlisted": int(len(U.unlisted)),
            "n_rows_merged": U.n_rows_merged, "r_dup": U.r_dup, "n_F_inf": int(np.sum(~np.isfinite(F))),
            "n_candidates": fs["n_cand"], "n_sources": fs["n_kept"], "cond": fs["cond"], "const": fs["const"],
            "degenerate": fs["degenerate"], "n_degenerate": int(np.sum(fs["degenerate"])),
            "own_move_px": np.column_stack([fs["own_x"] - star_x, fs["own_y"] - star_y]),
            "n_updates": fs["n_updates"], "projected": fs["projected"], "converged": fs["converged"]}
    info.update(_iteration0_counts("star", fs))
    # diagnostics of r5d 1.3 / A.11 (run.log): distance bins of other known sources, paired-row offsets, rows
    # near the paired row, detections near the r_u boundary
    info["neighbour_bins"] = iteration0_neighbour_bins(star_x, star_y, own_src, U.x, U.y, prof.fwhm, 1.0 / a)
    if len(star_row):
        info["paired_row_offset_px"] = np.hypot(rows_x[star_row] - star_x, rows_y[star_row] - star_y)
        info["n_stars_other_row_near_paired"] = _rows_near(rows_x, rows_y, star_row, 0.1 * prof.fwhm,
                                                           0.5 * prof.fwhm)
    else:
        info["paired_row_offset_px"] = np.zeros(0)
        info["n_stars_other_row_near_paired"] = 0
    info["n_det_near_r_u"] = int(np.sum((U.det_row_dist >= 0.8 * U.r_u) & (U.det_row_dist <= 1.2 * U.r_u)))
    t_det = t_fit
    if detection_fits:
        fd = fit_iteration0_boxes(image, sigma_sky, zero_w, np.rint(det_x).astype(np.int64),
                                  np.rint(det_y).astype(np.int64), det_x, det_y, U.det_src, U.x, U.y, F, prof,
                                  half=half)
        info["neighbour_flux"] = fd["own"]
        info["n_degenerate_det"] = int(np.sum(fd["degenerate"]))
        info.update(_iteration0_counts("det", fd))
        t_det = time.perf_counter()
    info["seconds"] = dict(profile=t_prof - t0, star_fits=t_fit - t_prof, detection_fits=t_det - t_fit,
                           total=time.perf_counter() - t0)
    mv = np.hypot(info["own_move_px"][:, 0], info["own_move_px"][:, 1]) if len(star_x) else np.zeros(1)
    logger.info(
        "iteration 0 (r5d): %s (%.4f, %.4f), %d stars, known sources %d (rows merged %d, unlisted %d, F inf %d), "
        "kept neighbours/box median %.0f max %d, degenerate %d, own move px median %.3f p99 %.3f max %.3f, "
        "projected %d (edge %d / back %d), not converged %d (A<=0 %d / 5 updates %d), near-centre bins %s, "
        "stars with another row 0.1-0.5 FWHM from the paired row %d, detections at 0.8-1.2 r_u %d, %.2f s",
        info["profile"], info["moffat_fwhm"], info["moffat_beta"], len(star_x), info["n_known"],
        info["n_rows_merged"], info["n_unlisted"], info["n_F_inf"],
        float(np.median(fs["n_kept"])) if len(star_x) else 0.0, int(np.max(fs["n_kept"])) if len(star_x) else 0,
        info["n_degenerate"], float(np.median(mv)), float(np.percentile(mv, 99)), float(np.max(mv)),
        info["n_star_projected"], info["n_star_projected_at_edge"], info["n_star_projected_returned"],
        info["n_star_not_converged"], info["n_star_own_nonpositive"], info["n_star_iterations_exhausted"],
        [(b["lo"], b["hi"], b["n_stars"], round(b["inv_a_share"], 4)) for b in info["neighbour_bins"]],
        info["n_stars_other_row_near_paired"], info["n_det_near_r_u"], info["seconds"]["total"])
    return a, info


def _iteration0_counts(kind: str, f: dict) -> dict:
    """Counts of r5d A.3 / A.11 for one kind of box ("star" or "det")."""
    nc = ~f["converged"]
    return {f"n_{kind}_projected": int(np.sum(f["projected"])),
            f"n_{kind}_projected_at_edge": int(np.sum(f["projected"] & f["at_edge"])),
            f"n_{kind}_projected_returned": int(np.sum(f["projected"] & ~f["at_edge"])),
            f"n_{kind}_not_converged": int(np.sum(nc)),
            f"n_{kind}_own_nonpositive": int(np.sum(nc & f["own_nonpositive"])),
            f"n_{kind}_iterations_exhausted": int(np.sum(nc & ~f["own_nonpositive"]))}


def _rows_near(rows_x, rows_y, star_row, r_lo: float, r_hi: float) -> int:
    """Training stars whose paired row has another list row at r_lo < distance <= r_hi."""
    from scipy.spatial import cKDTree

    if len(rows_x) < 2:
        return 0
    tree = cKDTree(np.column_stack([rows_x, rows_y]))
    n = 0
    for r in star_row:
        near = [j for j in tree.query_ball_point([rows_x[r], rows_y[r]], r=r_hi) if j != r]
        dd = np.hypot(rows_x[near] - rows_x[r], rows_y[near] - rows_y[r]) if near else np.zeros(0)
        n += int(np.any((dd > r_lo) & (dd <= r_hi)))
    return n


# ---------------------------------------------------------------------------
# Step 5: Per-star centered cutouts via photutils EPSFBuilder
# ---------------------------------------------------------------------------


#: Zero padding (px per side) of the Fourier sub-pixel shift of a stamp (design v2.5 3rd ed. 6.5:
#: the shift is below 0.5 px, so 16 px is enough; 0.38 ms instead of 1.0 ms per 61 px cutout).
FOURIER_PAD = 16


def fourier_shift(img: np.ndarray, sx: float, sy: float, pad: int = FOURIER_PAD) -> np.ndarray:
    """Content of ``img`` moved by (sx, sy) px -- out(p) = img(p - s) -- by the Fourier shift
    theorem on the image zero-padded by ``pad`` px on every side, cut back to its shape (real
    part, float64).

    This registers the PSF training stamps (design v2.5 3rd ed. 5.5): Lanczos-3, whose
    frequency response exceeds 1 at middle frequencies, sharpened the mean registered stamp
    (peak +0.23 % at FWHM 3.7-4.5 px; this shift: -0.03 % to 0.00 %).
    """
    from scipy import fft as sfft

    img = np.asarray(img, dtype=np.float64)
    n0, n1 = img.shape
    m0, m1 = n0 + 2 * int(pad), n1 + 2 * int(pad)
    padded = np.zeros((m0, m1))
    padded[pad:pad + n0, pad:pad + n1] = img
    fy = np.fft.fftfreq(m0)[:, None]
    if m0 % 2 and m1 % 2:
        # odd sizes: no Nyquist bin, so the real transform is the complex one exactly
        fx = np.fft.rfftfreq(m1)[None, :]
        spec = sfft.rfft2(padded, workers=1)
        spec *= np.exp(-2j * np.pi * fy * float(sy)) * np.exp(-2j * np.pi * fx * float(sx))
        out = sfft.irfft2(spec, s=(m0, m1), workers=1)
    else:
        fx = np.fft.fftfreq(m1)[None, :]
        spec = sfft.fft2(padded, workers=1)
        spec *= np.exp(-2j * np.pi * (fx * float(sx) + fy * float(sy)))
        out = sfft.ifft2(spec, workers=1).real
    return out[pad:pad + n0, pad:pad + n1]


def _gw_centroid_vec(
    cutouts: np.ndarray, init_xy: np.ndarray,
    fwhm_pix: float, max_iters: int = 8, tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Vectorized iterative Gaussian-weighted centroid.

    Numerical equivalent of SExtractor's WIN moments / Stetson's
    centroider: at each iteration, weight the (clipped-to-non-negative)
    pixel data by a Gaussian centered at the current estimate, then
    update the center to the weighted first moment. Converges in 3-5
    iterations on well-resolved stars; max_iters=8 + tol=1e-3 is a
    generous backstop that finishes in <1 ms per star at any star count.

    Compared to photutils' EPSFBuilder this is ~100x faster for the
    same per-star center quality (median 0.05 pix off, p95 0.08 pix,
    no outliers > 0.5 pix on T01829), because it avoids:

    - building + discarding the averaged ePSF (~12 s/band)
    - 30 outer iterations of scipy.optimize.least_squares per star
      against a FittableImageModel (~40 s/band at 320 stars)

    Parameters
    ----------
    cutouts : (N, H, W) float -- per-star cutouts (background-subtracted).
        Negative pixels are clipped to zero internally so noise + sky
        residuals don't contribute to the weighted moment.
    init_xy : (N, 2) -- starting (x, y) in CUTOUT coordinates. We use
        the cutout geometric center as the starting point.
    fwhm_pix : Gaussian weight FWHM, pixels. ~PSF FWHM is a good choice
        (the auto-FWHM mode finder's output works well in practice).

    Returns
    -------
    cx, cy, n_iters_used : per-star (x, y) center in CUTOUT coordinates,
        plus the actual number of iterations executed (early-exit on
        convergence).
    """
    N, H, W = cutouts.shape
    sigma = float(fwhm_pix) / 2.354820045
    inv_two_sig2 = 1.0 / (2.0 * sigma * sigma)
    yy, xx = np.mgrid[:H, :W].astype(np.float64)  # (H, W)

    cx = init_xy[:, 0].astype(np.float64).copy()
    cy = init_xy[:, 1].astype(np.float64).copy()
    data = np.maximum(cutouts.astype(np.float64), 0.0)  # (N, H, W)

    n_iter_used = 0
    for k in range(max_iters):
        n_iter_used = k + 1
        dx = xx[None, :, :] - cx[:, None, None]
        dy = yy[None, :, :] - cy[:, None, None]
        w = np.exp(-(dx * dx + dy * dy) * inv_two_sig2)
        wd = w * data
        s = wd.sum(axis=(1, 2))
        s_safe = np.where(s > 0, s, 1.0)
        cx_new = (wd * xx[None, :, :]).sum(axis=(1, 2)) / s_safe
        cy_new = (wd * yy[None, :, :]).sum(axis=(1, 2)) / s_safe
        # Stars with all-zero weighted data keep their previous center.
        cx_new = np.where(s > 0, cx_new, cx)
        cy_new = np.where(s > 0, cy_new, cy)
        max_delta = float(np.max(np.abs(np.concatenate([cx_new - cx, cy_new - cy]))))
        cx, cy = cx_new, cy_new
        if max_delta < tol:
            break

    return cx, cy, n_iter_used


def _extract_centered_stamps_gw(
    image: np.ndarray, bad_mask: np.ndarray,
    star_x: np.ndarray, star_y: np.ndarray,
    *,
    cutout_size: int, psf_size: int, fwhm_pix: float,
    star_local_bkg_subtract: bool,
    star_local_bkg_annulus_rin_pix: float,
    star_local_bkg_annulus_rout_pix: float,
    star_local_bkg_sigma: float, star_local_bkg_minpix: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fast path: vectorized GW centroid + Fourier shift (:func:`fourier_shift`).

    Same return signature as :func:`extract_centered_stamps`. Drops the
    photutils.NDData / EPSFStars roundtrip in favor of direct numpy
    slicing, and uses :func:`_gw_centroid_vec` instead of EPSFBuilder.
    """
    image = np.asarray(image, dtype=np.float32)
    bad_mask = np.asarray(bad_mask, dtype=bool)
    H, W = image.shape
    half_in = int(cutout_size) // 2
    crop_origin = (int(cutout_size) - int(psf_size)) // 2

    # Step 1: extract initial cutouts at SEP integer centers.
    img_orig_x = np.round(star_x).astype(np.int64)
    img_orig_y = np.round(star_y).astype(np.int64)
    in_frame = (
        (img_orig_x - half_in >= 0) & (img_orig_x + half_in + 1 <= W) &
        (img_orig_y - half_in >= 0) & (img_orig_y + half_in + 1 <= H)
    )
    keep_initial = np.where(in_frame)[0]
    if len(keep_initial) == 0:
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))
    cutouts = np.empty(
        (len(keep_initial), 2 * half_in + 1, 2 * half_in + 1),
        dtype=np.float32,
    )
    has_bad = np.zeros(len(keep_initial), dtype=bool)
    for k, i in enumerate(keep_initial):
        ix = int(img_orig_x[i]); iy = int(img_orig_y[i])
        cutouts[k] = image[iy - half_in:iy + half_in + 1,
                           ix - half_in:ix + half_in + 1]
        has_bad[k] = bad_mask[iy - half_in:iy + half_in + 1,
                              ix - half_in:ix + half_in + 1].any()
    keep_clean = ~has_bad
    cutouts = cutouts[keep_clean]
    keep_initial = keep_initial[keep_clean]
    n_kept = len(keep_initial)
    if n_kept == 0:
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))

    # Step 2: per-cutout local annulus background (used both for
    # centering input and for the final shifted cutout).
    if star_local_bkg_subtract:
        bkg_arr = np.empty(n_kept, dtype=np.float64)
        for k, i in enumerate(keep_initial):
            ix = int(img_orig_x[i]); iy = int(img_orig_y[i])
            bkg_arr[k] = _annulus_median(
                image, ix, iy,
                star_local_bkg_annulus_rin_pix, star_local_bkg_annulus_rout_pix,
                star_local_bkg_sigma, star_local_bkg_minpix, bad_mask,
            )
        good_bkg = np.isfinite(bkg_arr)
        cutouts = cutouts.copy()
        cutouts[good_bkg] -= bkg_arr[good_bkg, None, None].astype(np.float32)

    # Step 3: vectorized centroid (start from cutout geometric center).
    init_xy = np.full((n_kept, 2), float(half_in), dtype=np.float64)
    cx_local, cy_local, n_iters = _gw_centroid_vec(
        cutouts, init_xy, fwhm_pix=fwhm_pix, max_iters=8, tol=1e-3,
    )
    fitted_x_all = cx_local - half_in + img_orig_x[keep_initial].astype(np.float64)
    fitted_y_all = cy_local - half_in + img_orig_y[keep_initial].astype(np.float64)

    # Step 4: re-extract a clean cutout at integer-rounded fitted center,
    # Fourier-shift (fourier_shift) to put the sub-pixel center at the cutout
    # center, then crop to psf_size and renormalize. Reuses the local
    # annulus background from step 2 when the integer center didn't
    # change (which is the common case after sub-pixel-only refinement).
    out_stamps: list[np.ndarray] = []
    out_fx: list[float] = []
    out_fy: list[float] = []
    out_keep: list[int] = []
    for k, i in enumerate(keep_initial):
        cx_img = float(fitted_x_all[k])
        cy_img = float(fitted_y_all[k])
        if not (np.isfinite(cx_img) and np.isfinite(cy_img)):
            continue
        ix = int(round(cx_img)); iy = int(round(cy_img))
        dx = cx_img - ix
        dy = cy_img - iy
        y0c = iy - half_in; y1c = iy + half_in + 1
        x0c = ix - half_in; x1c = ix + half_in + 1
        if y0c < 0 or x0c < 0 or y1c > H or x1c > W:
            continue
        if bad_mask[y0c:y1c, x0c:x1c].any():
            continue
        cut = image[y0c:y1c, x0c:x1c].astype(np.float32, copy=True)
        if star_local_bkg_subtract:
            if (
                os.environ.get("TRACT7DT_BENCH_DISABLE_PSF_BKG_REUSE") != "1"
                and ix == int(img_orig_x[i])
                and iy == int(img_orig_y[i])
            ):
                local_bkg = bkg_arr[k]
            else:
                local_bkg = _annulus_median(
                    image, ix, iy,
                    star_local_bkg_annulus_rin_pix, star_local_bkg_annulus_rout_pix,
                    star_local_bkg_sigma, star_local_bkg_minpix, bad_mask,
                )
            if np.isfinite(local_bkg):
                cut = cut - np.float32(local_bkg)
        shifted = fourier_shift(cut, -dx, -dy)
        crop = shifted[crop_origin:crop_origin + psf_size,
                       crop_origin:crop_origin + psf_size]
        s = float(crop.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        out_stamps.append((crop / s).astype(np.float32))
        out_fx.append(cx_img); out_fy.append(cy_img)
        out_keep.append(int(i))

    if not out_stamps:
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))
    logger.debug(
        "gw centroid: %d stars, converged in %d iter(s)",
        n_kept, n_iters,
    )
    return (
        np.stack(out_stamps, axis=0),
        np.asarray(out_fx, dtype=np.float64),
        np.asarray(out_fy, dtype=np.float64),
        np.asarray(out_keep, dtype=np.int64),
    )


def _extract_centered_stamps_epsfbuilder(
    image: np.ndarray, sigma: np.ndarray, bad_mask: np.ndarray,
    star_x: np.ndarray, star_y: np.ndarray,
    *,
    cutout_size: int, psf_size: int,
    recenter_boxsize: int, recenter_maxiters: int, sigma_clip: float,
    star_local_bkg_subtract: bool,
    star_local_bkg_annulus_rin_pix: float,
    star_local_bkg_annulus_rout_pix: float,
    star_local_bkg_sigma: float, star_local_bkg_minpix: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Legacy slow path: photutils EPSFBuilder centres + Fourier shift.

    Kept for parity with the pre-Apr-2026 centres (the registration is now the Fourier shift
    of the default path, so its stamps no longer reproduce those outputs exactly; use
    ``centering_method: "epsfbuilder"`` to opt in). The per-star fit
    centers from EPSFBuilder differ from the GW centroid by median
    0.05 pix / max ~0.3 pix on real data; the resulting polynomial PSF
    differs by <0.1% peak amplitude. New deployments should use the
    default ``"gw_centroid"`` path.
    """
    from astropy.nddata import NDData, StdDevUncertainty
    from astropy.stats import SigmaClip
    from astropy.table import Table
    from photutils.psf import EPSFBuilder, extract_stars

    image = np.asarray(image, dtype=np.float32)
    sigma = np.asarray(sigma, dtype=np.float32)
    bad_mask = np.asarray(bad_mask, dtype=bool)

    err = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan).astype(np.float32)
    err_med = float(np.nanmedian(err)) if np.any(np.isfinite(err)) else 1.0
    err = np.where(np.isfinite(err), err, err_med).astype(np.float32)

    img_for_extract = image.copy()
    img_for_extract[bad_mask] = np.nan

    star_tbl = Table()
    star_tbl["x"] = np.asarray(star_x, dtype=np.float64)
    star_tbl["y"] = np.asarray(star_y, dtype=np.float64)

    nd = NDData(img_for_extract, uncertainty=StdDevUncertainty(err))
    try:
        stars = extract_stars(nd, star_tbl, size=int(cutout_size))
    except Exception as exc:
        logger.warning("extract_stars failed: %s", exc)
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))
    if stars is None or len(stars) == 0:
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))

    if star_local_bkg_subtract:
        stars_bg, _ = _subtract_local_annulus_background(
            stars, image, bad_mask,
            rin=star_local_bkg_annulus_rin_pix,
            rout=star_local_bkg_annulus_rout_pix,
            sigma=star_local_bkg_sigma,
            minpix=star_local_bkg_minpix,
        )
        if stars_bg is not None and len(stars_bg) > 0:
            stars = stars_bg

    sigclip = SigmaClip(sigma=float(sigma_clip)) if sigma_clip > 0 else SigmaClip(sigma=np.inf)
    builder = EPSFBuilder(
        oversampling=1,
        maxiters=int(recenter_maxiters),
        recentering_boxsize=int(recenter_boxsize),
        recentering_maxiters=30,
        sigma_clip=sigclip,
        progress_bar=False,
    )
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*fitting region extends beyond.*",
                module="photutils.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=".*cannot be fit because.*",
                module="photutils.*",
            )
            _epsf, fitted_stars = builder(stars)
    except Exception as exc:
        logger.warning("EPSFBuilder failed: %s", exc)
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))

    H, W = image.shape
    half_in = int(cutout_size) // 2
    crop_origin = (int(cutout_size) - int(psf_size)) // 2

    stamps_list: list[np.ndarray] = []
    fx: list[float] = []
    fy: list[float] = []
    keep_idx: list[int] = []
    for i, fs in enumerate(fitted_stars):
        try:
            cx_img, cy_img = float(fs.center[0]), float(fs.center[1])
        except Exception:
            continue
        if not (np.isfinite(cx_img) and np.isfinite(cy_img)):
            continue
        ix = int(round(cx_img)); iy = int(round(cy_img))
        dx = cx_img - ix
        dy = cy_img - iy
        y0c = iy - half_in; y1c = iy + half_in + 1
        x0c = ix - half_in; x1c = ix + half_in + 1
        if y0c < 0 or x0c < 0 or y1c > H or x1c > W:
            continue
        cut = image[y0c:y1c, x0c:x1c].astype(np.float32).copy()
        bad_cut = bad_mask[y0c:y1c, x0c:x1c]
        if bad_cut.any():
            continue
        local_bkg = _annulus_median(
            image, ix, iy,
            star_local_bkg_annulus_rin_pix, star_local_bkg_annulus_rout_pix,
            star_local_bkg_sigma, star_local_bkg_minpix, bad_mask,
        )
        if np.isfinite(local_bkg):
            cut = cut - np.float32(local_bkg)
        shifted = fourier_shift(cut, -dx, -dy)
        crop = shifted[crop_origin:crop_origin + psf_size,
                       crop_origin:crop_origin + psf_size]
        s = float(crop.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        stamps_list.append((crop / s).astype(np.float32))
        fx.append(cx_img); fy.append(cy_img)
        keep_idx.append(i)

    if not stamps_list:
        return (np.zeros((0, psf_size, psf_size), dtype=np.float32),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.int64))
    return (
        np.stack(stamps_list, axis=0),
        np.asarray(fx, dtype=np.float64),
        np.asarray(fy, dtype=np.float64),
        np.asarray(keep_idx, dtype=np.int64),
    )


def extract_centered_stamps(
    image: np.ndarray,
    sigma: np.ndarray,
    bad_mask: np.ndarray,
    star_x: np.ndarray,
    star_y: np.ndarray,
    *,
    cutout_size: int = 61,
    psf_size: int = 55,
    recenter_boxsize: int = 9,
    recenter_maxiters: int = 30,
    sigma_clip: float = 5.0,
    star_local_bkg_subtract: bool = True,
    star_local_bkg_annulus_rin_pix: float = 20.0,
    star_local_bkg_annulus_rout_pix: float = 30.0,
    star_local_bkg_sigma: float = 3.0,
    star_local_bkg_minpix: int = 30,
    centering_method: str = "gw_centroid",
    fwhm_pix: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-star sub-pixel-centered cutouts.

    Returns ``(stamps, fitted_x, fitted_y, kept_indices)`` where:

    - ``stamps`` : ``(n_kept, psf_size, psf_size)`` sub-pixel-centered +
      sky-subtracted + sum-1-normalized cutouts.
    - ``fitted_x``, ``fitted_y`` : per-star sub-pixel-fit centers in
      full-image coordinates.
    - ``kept_indices`` : indices into the input ``star_x/y`` arrays of
      stars that were successfully centered + cropped (some get dropped
      for edge proximity, bad pixels, or zero/NaN sum).

    Parameters
    ----------
    centering_method : str
        Sub-pixel centering algorithm:
          * ``"gw_centroid"`` (default): vectorized iterative
            Gaussian-weighted centroid (see :func:`_gw_centroid_vec`).
            ~100x faster than ``"epsfbuilder"`` per band, with
            sub-percent change to the polynomial PSF model.
          * ``"epsfbuilder"``: photutils' ``EPSFBuilder`` iterative
            recentering. Kept for parity with pre-Apr-2026 outputs.
        Unknown values raise ``ValueError``.
    fwhm_pix : float
        Gaussian-weight FWHM passed to the GW centroider. Should be
        approximately the band's PSF FWHM; the auto-FWHM mode finder's
        output is a good choice. Ignored when
        ``centering_method == "epsfbuilder"``.
    """
    method = str(centering_method).strip().lower()
    if method == "gw_centroid":
        return _extract_centered_stamps_gw(
            image, bad_mask, star_x, star_y,
            cutout_size=cutout_size, psf_size=psf_size,
            fwhm_pix=fwhm_pix,
            star_local_bkg_subtract=star_local_bkg_subtract,
            star_local_bkg_annulus_rin_pix=star_local_bkg_annulus_rin_pix,
            star_local_bkg_annulus_rout_pix=star_local_bkg_annulus_rout_pix,
            star_local_bkg_sigma=star_local_bkg_sigma,
            star_local_bkg_minpix=star_local_bkg_minpix,
        )
    elif method == "epsfbuilder":
        return _extract_centered_stamps_epsfbuilder(
            image, sigma, bad_mask, star_x, star_y,
            cutout_size=cutout_size, psf_size=psf_size,
            recenter_boxsize=recenter_boxsize,
            recenter_maxiters=recenter_maxiters,
            sigma_clip=sigma_clip,
            star_local_bkg_subtract=star_local_bkg_subtract,
            star_local_bkg_annulus_rin_pix=star_local_bkg_annulus_rin_pix,
            star_local_bkg_annulus_rout_pix=star_local_bkg_annulus_rout_pix,
            star_local_bkg_sigma=star_local_bkg_sigma,
            star_local_bkg_minpix=star_local_bkg_minpix,
        )
    else:
        raise ValueError(
            f"Unknown centering_method={centering_method!r}; "
            "expected 'gw_centroid' or 'epsfbuilder'"
        )


def _annulus_median(
    image: np.ndarray, ix: int, iy: int,
    rin: float, rout: float, sigma: float, minpix: int,
    bad_mask: np.ndarray,
) -> float:
    """Sigma-clipped median in an annular region around (ix, iy)."""
    H, W = image.shape
    ri = int(np.ceil(rout))
    y0 = max(0, iy - ri); y1 = min(H, iy + ri + 1)
    x0 = max(0, ix - ri); x1 = min(W, ix + ri + 1)
    sub = image[y0:y1, x0:x1]
    sub_bad = bad_mask[y0:y1, x0:x1]
    yy, xx = np.mgrid[:sub.shape[0], :sub.shape[1]]
    cy = iy - y0; cx = ix - x0
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    annulus = (r2 >= rin * rin) & (r2 <= rout * rout) & (~sub_bad) & np.isfinite(sub)
    vals = sub[annulus]
    if vals.size < minpix:
        return float("nan")
    # Sigma-clip
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    if mad <= 0:
        return med
    keep = (vals >= med - sigma * mad) & (vals <= med + sigma * mad)
    if keep.sum() < minpix:
        return med
    return float(np.median(vals[keep]))


def _subtract_local_annulus_background(
    stars, image, bad_mask, *,
    rin: float, rout: float, sigma: float, minpix: int,
):
    """Subtract a local annulus background from each star in a
    ``photutils.psf.EPSFStars`` collection, in-place. Returns the
    same collection and a list of (per-star) background stats."""
    stats: list[dict[str, float]] = []
    H, W = image.shape
    for s in stars:
        try:
            x_img = float(s.center[0]); y_img = float(s.center[1])
        except Exception:
            stats.append({"bkg": float("nan"), "ok": 0})
            continue
        ix = int(round(x_img)); iy = int(round(y_img))
        bkg = _annulus_median(
            image, ix, iy, rin, rout, sigma, minpix, bad_mask,
        )
        if np.isfinite(bkg):
            s.data[:] = (s.data - np.float32(bkg)).astype(s.data.dtype)
            stats.append({"bkg": float(bkg), "ok": 1})
        else:
            stats.append({"bkg": float("nan"), "ok": 0})
    return stars, stats


# ---------------------------------------------------------------------------
# Step 6: Per-pixel polynomial fit + PSFEx-style outlier rejection loop
# ---------------------------------------------------------------------------


def _fit_polynomial_one_iter(
    stars_x: np.ndarray, stars_y: np.ndarray,
    stamps: np.ndarray,
    image_shape: tuple[int, int],
    polynomial_order: int = 2,
    rcond: float = 1e-9,
) -> tuple[np.ndarray, float, float, float, float]:
    """One pass of the per-pixel polynomial fit.

    Per-pixel least squares of the stamp values on the spatial
    polynomial basis, parameterized by polynomial order.

    Returns ``(coefs, x_origin, x_scale, y_origin, y_scale)`` with
    ``coefs`` of shape ``(H_psf, W_psf, n_terms)``.
    """
    H, W = image_shape
    x_origin = 0.5 * W; y_origin = 0.5 * H
    x_scale = 0.5 * W; y_scale = 0.5 * H
    X = (stars_x - x_origin) / x_scale
    Y = (stars_y - y_origin) / y_scale
    terms = terms_for_order(int(polynomial_order))
    n_terms = len(terms)
    n_stars = stamps.shape[0]
    H_psf, W_psf = stamps.shape[1], stamps.shape[2]
    # Design matrix A: (n_stars, n_terms)
    A = np.empty((n_stars, n_terms), dtype=np.float64)
    for t, (m, n) in enumerate(terms):
        A[:, t] = (X ** m) * (Y ** n)
    # Per-pixel right-hand side: (n_stars, H_psf*W_psf)
    Y_mat = stamps.reshape(n_stars, H_psf * W_psf).astype(np.float64)
    # Solve: coefs (n_terms, H_psf*W_psf) = pinv(A) @ Y_mat
    coefs_flat, _resid, _rank, _sv = np.linalg.lstsq(A, Y_mat, rcond=rcond)
    coefs = coefs_flat.T.reshape(H_psf, W_psf, n_terms).astype(np.float32)
    return coefs, float(x_origin), float(x_scale), float(y_origin), float(y_scale)


def _per_star_chi2(
    coefs: np.ndarray, x_origin: float, x_scale: float,
    y_origin: float, y_scale: float, polynomial_order: int,
    stars_x: np.ndarray, stars_y: np.ndarray, stamps: np.ndarray,
) -> np.ndarray:
    """For each star, evaluate the polynomial PSF at its (x, y) and
    compute the per-pixel chi^2 ~ sum (stamp - model)^2 (per-stamp,
    summed over pixels). We don't have per-pixel weights here, so this
    is a simple sum-of-squares used for outlier rejection only.
    """
    n_stars = stamps.shape[0]
    H_psf, W_psf, n_terms = coefs.shape
    X = (stars_x - x_origin) / x_scale
    Y = (stars_y - y_origin) / y_scale
    terms = terms_for_order(int(polynomial_order))
    A = np.empty((n_stars, n_terms), dtype=np.float64)
    for t, (m, n) in enumerate(terms):
        A[:, t] = (X ** m) * (Y ** n)
    # coefs has shape (H_psf, W_psf, n_terms). For each pixel of the PSF
    # the model at star i is sum_t coefs[h, w, t] * A[i, t]. Reshape so
    # the matmul A @ coefs_flat.T gives (n_stars, H_psf * W_psf).
    coefs_flat = coefs.reshape(H_psf * W_psf, n_terms).astype(np.float64)  # (P, T)
    model_flat = A @ coefs_flat.T  # (n_stars, T) @ (T, P) -> (n_stars, P)
    diff = stamps.reshape(n_stars, -1).astype(np.float64) - model_flat
    return np.sum(diff * diff, axis=1)


def _replace_model_wings_with_median(
    psf: PsfPolynomial,
    stamps: np.ndarray,
    accepted: "np.ndarray | None",
    fwhm_pix: float,
    band: str,
    blend_start_fwhm: float = 2.5,
    blend_width_pix: float = 4.0,
) -> PsfPolynomial:
    """Replace the polynomial model's wings with the pixelwise MEDIAN of the
    training stamps, blended smoothly beyond ~``blend_start_fwhm`` x FWHM.

    WHY: the per-pixel least-squares stack inherits the pixelwise MEAN of
    whatever contaminates the training stamps. In crowded fields the mean of
    (sub-threshold, unmaskable) neighbour flux puts a flat false pedestal in
    the model wings -- measured on T08285 m525: model ring(18-27px) 9.3e-5
    vs 8.2e-6 for the pixelwise median, an 11x excess. Every fitted source
    then over-subtracts a fake halo scaled by its own flux, producing a
    one-sided faint-end flux deficit (the diagonal tail in zp_vs_mag plots),
    a -0.05 mag faint-end drift, and a spurious spatial ZP term. The MEDIAN
    over stars is robust to that contamination (any given wing pixel is
    contaminated in a minority of stars), and the TRUE wing's spatial
    variation at r > 2.5 FWHM is far below the contamination floor, so a
    spatially-constant clean wing is strictly better than a polynomial
    pedestal. The constant term blends to the median wing, the spatial terms
    are damped to zero over the same blend (so they cannot re-introduce the
    pedestal), and the constant term is renormalized to its pre-surgery
    integral so the model flux convention is preserved exactly.
    """
    st = stamps[accepted] if accepted is not None else stamps
    if st.shape[0] < 20:
        logger.info("[%s] wing surgery skipped: only %d stars (<20)", band, st.shape[0])
        return psf
    med = np.median(st.astype(np.float64), axis=0)
    coefs = psf.coefs.astype(np.float64).copy()
    H_psf, W_psf, n_terms = coefs.shape
    yy, xx = np.mgrid[0:H_psf, 0:W_psf]
    r = np.hypot(yy - (H_psf - 1) / 2.0, xx - (W_psf - 1) / 2.0)
    fw = float(fwhm_pix) if np.isfinite(fwhm_pix) and fwhm_pix > 0 else 5.0
    r0 = max(6.0, blend_start_fwhm * fw)
    w = np.clip((r - r0) / max(blend_width_pix, 1e-6), 0.0, 1.0)
    ring = (r >= 18) & (r <= 27)
    ring_before = float(coefs[:, :, 0][ring].mean()) if ring.any() else float("nan")
    c0 = coefs[:, :, 0]
    target_sum = float(c0.sum())
    c0_new = (1.0 - w) * c0 + w * med
    s = float(c0_new.sum())
    if s > 0 and np.isfinite(s):
        c0_new *= target_sum / s
    coefs[:, :, 0] = c0_new
    if n_terms > 1:
        coefs[:, :, 1:] *= (1.0 - w)[:, :, None]
    ring_after = float(coefs[:, :, 0][ring].mean()) if ring.any() else float("nan")
    logger.info(
        "[%s] wing surgery: blend r0=%.1fpx (%.1f x FWHM %.2fpx), "
        "ring(18-27px) %.2e -> %.2e (median-stamp ref %.2e)",
        band, r0, blend_start_fwhm, fw,
        ring_before, ring_after,
        float(med[ring].mean()) if ring.any() else float("nan"),
    )
    return PsfPolynomial(
        coefs=coefs.astype(np.float32),
        x_origin=psf.x_origin, x_scale=psf.x_scale,
        y_origin=psf.y_origin, y_scale=psf.y_scale,
        H_psf=int(H_psf), W_psf=int(W_psf),
    )


def fit_polynomial_with_outlier_rejection(
    stars_x: np.ndarray, stars_y: np.ndarray, stamps: np.ndarray,
    image_shape: tuple[int, int],
    polynomial_order: int = 2,
    max_outer_iters: int = 4,
    kappa: float = 3.0,
    rcond: float = 1e-9,
) -> tuple[PsfPolynomial, np.ndarray, int, np.ndarray]:
    """Iterative polynomial fit with PSFEx-style outlier rejection.

    Loop:
      1. Fit per-pixel polynomial from current accepted stars.
      2. Compute per-star chi^2 (sum (stamp-model)^2 over pixels).
      3. Take chi = sqrt(chi2). Compute median + sigma (MAD-based).
      4. Reject stars with chi > median + kappa * sigma.
      5. Iterate up to ``max_outer_iters``; stop early if no rejections.

    Returns ``(psf_polynomial, kept_mask, n_iters_used, final_chi2_per_star)``.
    """
    n_stars = stamps.shape[0]
    if n_stars == 0:
        raise ValueError("No stars to fit")
    H_psf, W_psf = stamps.shape[1], stamps.shape[2]

    accepted = np.ones(n_stars, dtype=bool)
    n_iters_used = 0
    final_chi2 = np.zeros(n_stars, dtype=np.float64)
    coefs = None; x_o = x_s = y_o = y_s = 0.0

    for it in range(max_outer_iters):
        n_iters_used = it + 1
        if accepted.sum() < n_terms_for_order(polynomial_order):
            logger.warning(
                "polynomial fit: only %d accepted stars (< %d required for "
                "order %d); stopping early",
                int(accepted.sum()), n_terms_for_order(polynomial_order),
                int(polynomial_order),
            )
            break
        coefs, x_o, x_s, y_o, y_s = _fit_polynomial_one_iter(
            stars_x[accepted], stars_y[accepted], stamps[accepted],
            image_shape=image_shape,
            polynomial_order=polynomial_order, rcond=rcond,
        )
        # Compute chi2 for ALL stars (incl. previously rejected ones, so
        # they can rejoin).
        chi2 = _per_star_chi2(
            coefs, x_o, x_s, y_o, y_s, polynomial_order,
            stars_x, stars_y, stamps,
        )
        chi = np.sqrt(np.maximum(chi2, 0.0))
        # Robust statistics on accepted stars only.
        chi_acc = chi[accepted]
        if len(chi_acc) < 3:
            final_chi2 = chi2
            break
        med = float(np.median(chi_acc))
        mad = float(np.median(np.abs(chi_acc - med))) * 1.4826
        if mad <= 0:
            final_chi2 = chi2
            break
        accepted_new = chi <= med + kappa * mad
        n_rejected_now = int((accepted & ~accepted_new).sum())
        n_recovered = int((~accepted & accepted_new).sum())
        accepted = accepted_new
        final_chi2 = chi2
        logger.info(
            "polynomial fit iter %d: kept=%d, rejected=%d, recovered=%d, "
            "chi median=%.3e, mad=%.3e",
            it + 1, int(accepted.sum()), n_rejected_now, n_recovered,
            med, mad,
        )
        if n_rejected_now == 0 and n_recovered == 0:
            break

    if coefs is None:
        raise RuntimeError("polynomial fit failed (no iterations completed)")

    psf = PsfPolynomial(
        coefs=coefs.astype(np.float32),
        x_origin=float(x_o), x_scale=float(x_s),
        y_origin=float(y_o), y_scale=float(y_s),
        H_psf=int(H_psf), W_psf=int(W_psf),
    )
    return psf, accepted, n_iters_used, final_chi2


# ---------------------------------------------------------------------------
# Top-level: build PSF polynomial for one band end-to-end
# ---------------------------------------------------------------------------


def build_psf_polynomial_from_stars_for_band(
    band: str,
    image: np.ndarray, sigma: np.ndarray, bad_mask: np.ndarray,
    sat_mask: np.ndarray,
    cfg: dict[str, Any],
    *,
    rows: TrainingRows,
    linear_limit: float,
) -> PsfFromStarsResult:
    """Run the full PSFEx-style pipeline for one band:
    SEP detect -> auto-FWHM mode -> training stars (:func:`select_psf_stars`: identity
    rows ``rows``, the band's saturation mask ``sat_mask``, ``linear_limit`` in image units)
    -> centered cutouts -> polynomial fit with outlier rejection -> return PsfPolynomial +
    diagnostics.
    """
    t0 = time.time()
    H, W = image.shape

    # 1. Detect SEP objects.
    objs = detect_sep_objects(
        image, sigma, bad_mask,
        thresh_sigma=cfg["thresh_sigma"], minarea=cfg["minarea"],
        deblend_nthresh=cfg["deblend_nthresh"], deblend_cont=cfg["deblend_cont"],
    )
    n_detected = len(objs)
    logger.info("[%s] sep.extract: %d raw detections", band, n_detected)
    if n_detected == 0:
        raise RuntimeError(f"[{band}] sep detected zero objects")

    x_all, y_all, fwhm_all, ellip_all, snr_all = _objs_with_quality_metrics(objs)

    # 2. PSFEx auto-FWHM mode finder (on detected objects passing
    #    a generous SNR pre-filter so noise spikes don't bias the mode).
    snr_pre = snr_all > 5.0
    if snr_pre.sum() >= 10:
        fwhm_for_mode = fwhm_all[snr_pre]
    else:
        fwhm_for_mode = fwhm_all
    fwhm_mode, fwhm_min, fwhm_max = compute_fwhmrange_psfex(
        fwhm_for_mode, sample_variability=cfg["sample_variability"],
    )
    logger.info(
        "[%s] PSFEx auto FWHM: mode=%.2f pix, range=[%.2f, %.2f]",
        band, fwhm_mode, fwhm_min, fwhm_max,
    )

    # 3. PSF training stars (design 1.5).
    selected_idx, cut_counts, _ = select_psf_stars(
        objs, image_shape=(H, W),
        fwhm_mode=fwhm_mode,
        min_snr=cfg["min_snr"],
        edge_pad_pix=cfg["edge_pad"] + cfg["cutout_size"] // 2,
        cutout_size=cfg["cutout_size"], psf_size=cfg["psf_size"],
        bad_mask=bad_mask, sat_mask=sat_mask, linear_limit=linear_limit, rows=rows,
    )
    logger.info(
        "[%s] PSF training stars (%s rows): %s -> kept %d",
        band, rows.origin, cut_counts, len(selected_idx),
    )
    if len(selected_idx) == 0:
        raise RuntimeError(f"[{band}] no PSF training star passed the rules: {cut_counts}")
    n_after_quality = len(selected_idx)

    sel_x = x_all[selected_idx]; sel_y = y_all[selected_idx]
    sel_fwhm = fwhm_all[selected_idx]; sel_ellip = ellip_all[selected_idx]
    sel_snr = snr_all[selected_idx]
    sel_flux = (np.asarray(objs["flux"], dtype=np.float64)[selected_idx]
                if "flux" in objs.dtype.names else np.zeros_like(sel_x))

    n_after_gaia = int(cut_counts["after_identity"])
    origin_arr = np.array([rows.origin] * len(sel_x), dtype="U16")

    # Cap stars to max_stars_per_band (keep highest SNR).
    max_stars = int(cfg["max_stars_per_band"])
    if len(sel_x) > max_stars:
        order = np.argsort(-sel_snr)[:max_stars]
        sel_x = sel_x[order]; sel_y = sel_y[order]
        sel_fwhm = sel_fwhm[order]; sel_ellip = sel_ellip[order]
        sel_snr = sel_snr[order]; sel_flux = sel_flux[order]
        origin_arr = origin_arr[order]
        logger.info("[%s] capped to top-%d by SNR", band, max_stars)

    centering_method = str(cfg.get("centering_method", "gw_centroid"))

    stamps, fitted_x, fitted_y, kept_extract = extract_centered_stamps(
        image, sigma, bad_mask, sel_x, sel_y,
        cutout_size=cfg["cutout_size"], psf_size=cfg["psf_size"],
        recenter_boxsize=cfg["recenter_boxsize"],
        recenter_maxiters=cfg["recenter_maxiters"],
        sigma_clip=cfg["sigma_clip"],
        star_local_bkg_subtract=cfg["star_local_bkg_subtract"],
        star_local_bkg_annulus_rin_pix=cfg["star_local_bkg_annulus_rin_pix"],
        star_local_bkg_annulus_rout_pix=cfg["star_local_bkg_annulus_rout_pix"],
        star_local_bkg_sigma=cfg["star_local_bkg_sigma"],
        star_local_bkg_minpix=cfg["star_local_bkg_minpix"],
        centering_method=centering_method,
        fwhm_pix=float(fwhm_mode) if np.isfinite(fwhm_mode) else 5.0,
    )
    n_extracted = len(kept_extract)
    logger.info("[%s] (legacy) centered+normalized: %d / %d stars",
                band, n_extracted, len(sel_x))
    if n_extracted < n_terms_for_order(cfg["polynomial_order"]):
        raise RuntimeError(
            f"[{band}] only {n_extracted} stars extracted; need at least "
            f"{n_terms_for_order(cfg['polynomial_order'])} for order "
            f"{cfg['polynomial_order']}")
    sel_fwhm_kept = sel_fwhm[kept_extract]; sel_ellip_kept = sel_ellip[kept_extract]
    sel_flux_kept = sel_flux[kept_extract]; origin_kept = origin_arr[kept_extract]
    psf, accepted_mask, n_iters, chi2_per_star = fit_polynomial_with_outlier_rejection(
        fitted_x, fitted_y, stamps, image_shape=(H, W),
        polynomial_order=cfg["polynomial_order"],
        max_outer_iters=cfg["max_outer_iters"], kappa=cfg["kappa"])
    if bool(cfg.get("wing_from_median", True)):
        psf = _replace_model_wings_with_median(
            psf, stamps, accepted_mask,
            fwhm_pix=float(fwhm_mode) if np.isfinite(fwhm_mode) else 5.0,
            band=band,
            blend_start_fwhm=float(cfg.get("wing_blend_start_fwhm", 2.5)),
        )
    n_used = int(accepted_mask.sum())
    logger.info("[%s] (legacy) polynomial fit DONE: %d / %d stars, %d iters",
                band, n_used, n_extracted, n_iters)
    return PsfFromStarsResult(
        band=band, psf_polynomial=psf,
        polynomial_order=int(cfg["polynomial_order"]),
        n_stars_detected=int(n_detected), n_stars_after_quality=int(n_after_quality),
        n_stars_after_gaia=int(n_after_gaia), n_stars_used_in_fit=int(n_used),
        fwhm_mode=float(fwhm_mode), fwhm_min=float(fwhm_min), fwhm_max=float(fwhm_max),
        star_x=fitted_x[accepted_mask], star_y=fitted_y[accepted_mask],
        star_chi2=chi2_per_star[accepted_mask], star_flux=sel_flux_kept[accepted_mask],
        star_fwhm=sel_fwhm_kept[accepted_mask], star_ellip=sel_ellip_kept[accepted_mask],
        star_origin=origin_kept[accepted_mask],
        star_centered_stamps=stamps[accepted_mask].astype(np.float32),
        n_outer_iters=int(n_iters),
        final_chi2_mean=float(chi2_per_star[accepted_mask].mean()) if n_used else float("nan"),
        final_chi2_max=float(chi2_per_star[accepted_mask].max()) if n_used else float("nan"),
        timing_sec=float(time.time() - t0),
        psf_normalization_used="sum",
    )


__all__ = [
    "PsfFromStarsResult",
    "detect_sep_objects",
    "compute_fwhmrange_psfex",
    "GAIA_ID_COLUMN",
    "TrainingRows",
    "psf_training_rows",
    "pair_detections_with_rows",
    "select_psf_stars",
    "FOURIER_PAD",
    "fourier_shift",
    "isolated_in_box",
    "sep_auto_flux",
    "empirical_star_stamp",
    "MoffatProfile",
    "GaussianProfile",
    "fit_moffat",
    "radial_profile",
    "KnownSources",
    "known_sources",
    "plausible_amplitude",
    "fit_iteration0_boxes",
    "iteration0_neighbour_bins",
    "star_noise_a",
    "box_sum_kappa",
    "psf0_star_noise",
    "neighbour_amplitude",
    "star_noise_t",
    "psf0_training_noise",
    "extract_centered_stamps",
    "fit_polynomial_with_outlier_rejection",
    "build_psf_polynomial_from_stars_for_band",
]
