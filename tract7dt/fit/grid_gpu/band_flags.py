"""Band-level output contract of the GPU catalog (plan v3 stage 1).

A source's fit can be healthy while one of its bands is not measured, or is
measured with a model that is known to be wrong there. These helpers make the
band level explicit:

* R05 -- a band cell with no information has no value
  (:func:`blank_nodata_band_cells`);
* R06 -- per-band reason bitmask ``FLAGS_{band}_fit`` and the count of usable
  bands ``n_bands_usable`` (:func:`compose_band_flags`);
* R07 -- where the band's PSF model is unphysical (:func:`psf_lobe_zone`);
* R11 -- saturation is a band-level exclusion (values blanked, bit 8), not a
  reason to drop the source from every band;
* R30 -- where part of the source's PSF falls on pixels without data in that
  band (the edge of the band's footprint), so the value leans on the PSF
  model over pixels nobody observed (:func:`psf_uncovered_fraction`).

A band value is usable when ``FLAGS_{band}_fit == 0`` and the source's
``fit_is_reliable`` is true.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ``FLAGS_{band}_fit`` bits.
FLAG_NO_DATA = 1      # FLUXERR undefined: no information in this band (R05)
FLAG_PSF_LOBE = 2     # PSF model has a negative lobe at the source (R07)
FLAG_ZP_EXTRAP = 4    # reserved: zero point extrapolated (R08)
FLAG_SATURATED = 8    # saturated pixel near the source in this band (R11)
FLAG_PARTIAL_COVERAGE = 16  # > PARTIAL_COVERAGE_MAX of the PSF on no-data pixels (R30)

# R30: a cell is marked when more than this fraction of the source's PSF
# (positive part, over the fit stamp) falls on no-data pixels of the band
# (non-finite, or outside the band's footprint; beyond the image counts too).
# Measured 2026-10-02 on the dev8 session-4 runs (anchors T08676/T08285 + the
# 21-tile request sample; Gaia-XP calibrators S/N >= 20 with no other flag,
# final MAG - XP; a development analysis script, not distributed): the
# median offset stays < 15 mmag at every fraction, but the error stops
# covering the scatter -- median |offset/MAGERR_MARG_TOT| relative to fully
# covered cells (n=1.83M) is x1.12 at 0-3% (n=13,566; the same near-edge
# excess at any fraction: the noise model's single SKYSIG per frame, stage
# 3b), x1.14-1.22 at 3-10%, x1.17-1.32 at 10-20%, x1.65-1.70 at 20-45%,
# x2.4-2.7 above 45%; at S/N >= 100 already x1.47 at 10-15%. The value is
# kept: with nearly all of the PSF unobserved its flux error is large
# (2026-09 request sample J082813 m425/m850: 99.9% unobserved, S/N 0.9-1.4)
# but the model, not the data, carries the measurement.
PARTIAL_COVERAGE_MAX = 0.1

# A real PSF is non-negative; the model stamp dipping below -0.2% of its peak
# marks it as broken there. Measured on dev7 T08676 (Gaia-XP calibrators,
# S/N>=20, final MAG - XP relative to the clean median; a development analysis
# script, not distributed): lobe < 0.2% -> +0.000 inside the PSF-
# star hull (n=69,107), +0.004 mag outside (n=4,551); 0.2-0.5% outside ->
# +0.056 (n=92); 0.5-1% -> +0.20; 1-5% -> +0.23..+0.30; >= 5% -> +0.70 mag
# (n=411); inside the hull where lobes occur (m850, 63 PSF stars) +0.06.
# The plan's first figure (5%, with the hull) left the 0.2-5% cells unmarked.
PSF_LOBE_FRAC = 0.002

# Per-band catalog columns that carry a value derived from the band's linear
# flux solve: "{prefix}{band}_fit". MAG/MAGERR/MAG_UL5 are derived from these
# later (compute_zp, or the zp-disabled block of run_grid_gpu_fit).
BAND_FLUX_COLUMN_PREFIXES = (
    "FLUX_", "FLUXERR_", "FLUXERR_CONF_", "FLUX_WIN_", "FLUXERR_TOT_",
    "FLUXERR_MARG_", "FLUX_MARG_FRAC_", "FLUXERR_MARG_TOT_",
)


def blank_nodata_band_cells(out: pd.DataFrame, bandnames) -> dict:
    """Output contract (REGISTER R05): a band cell with no information has no
    value.

    ``FLUXERR_{band}_fit`` is NaN exactly where the band's linear solve had no
    information about the source (non-finite or non-positive variance: the
    stamp lies entirely on zero-coverage padding, or the column is
    near-degenerate) or where the source has no fit at all. In such a cell the
    solver's exact-zero amplitude minus the grid's blank-response b_hat used
    to reach the catalog as ``FLUX = -b_hat`` (and ``FLUX_WIN = 0``,
    ``FLUX_MARG_FRAC = 0``), and ``compute_zp`` turned the positive ones into
    a finite MAG. Every per-band flux column of the cell is set to NaN
    in-place, so MAG is NaN as well.

    Returns ``{band: n}``, the number of blanked cells that still carried a
    finite ``FLUX_{band}_fit``.
    """
    n_blanked = {}
    for bn in bandnames:
        fe_col = f"FLUXERR_{bn}_fit"
        if fe_col not in out.columns:
            continue
        fe = pd.to_numeric(out[fe_col], errors="coerce").to_numpy(
            dtype=np.float64)
        nodata = ~np.isfinite(fe)
        fl_col = f"FLUX_{bn}_fit"
        if fl_col in out.columns:
            fl = pd.to_numeric(out[fl_col], errors="coerce").to_numpy(
                dtype=np.float64)
            n_blanked[bn] = int(np.count_nonzero(nodata & np.isfinite(fl)))
        if not np.any(nodata):
            continue
        for prefix in BAND_FLUX_COLUMN_PREFIXES:
            col = f"{prefix}{bn}_fit"
            if col in out.columns:
                out.loc[nodata, col] = np.nan
    return n_blanked


def psf_lobe_zone(psf, x, y, *, lobe_frac: float = PSF_LOBE_FRAC,
                  chunk: int = 4096) -> np.ndarray:
    """R07: True where the band's PSF model stamp at ``(x, y)`` has a negative
    lobe deeper than ``lobe_frac`` of its peak (or no positive peak at all).

    Mostly the extrapolated zone outside the PSF training stars' hull, where
    the spatial polynomial runs away (T08676: calibrators there 0.5-0.9 mag
    too faint in six bands; the 2026-09-04 21-tile delivery: 0.7-2.5 mag),
    but also sparsely sampled patches inside it. A forced PSF flux there is
    biased faint, so the band value is marked; the model itself is repaired in
    stage 3a. Positions that are not finite are never flagged.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    zone = np.zeros(x.shape, dtype=bool)
    idx = np.nonzero(np.isfinite(x) & np.isfinite(y))[0]
    if idx.size == 0:
        return zone
    from .psf_polynomial import evaluate_psf_at_batch_np
    for s in range(0, idx.size, int(chunk)):
        c = idx[s:s + int(chunk)]
        stamps = evaluate_psf_at_batch_np(psf, x[c], y[c])
        pk = stamps.max(axis=(1, 2)).astype(np.float64)
        mn = stamps.min(axis=(1, 2)).astype(np.float64)
        zone[c] = (pk <= 0.0) | (mn < -float(lobe_frac) * pk)
    return zone


def psf_uncovered_fraction(psf, nodata: np.ndarray, x, y, *,
                           chunk: int = 2048) -> np.ndarray:
    """R30: the fraction of the band's PSF at ``(x, y)`` that falls on
    pixels without data.

    ``nodata`` is the band's (cropped-frame) mask of pixels the fit cannot use
    for lack of data -- non-finite, or outside the band's footprint
    (``load_inputs`` rule) -- *before* saturation and the bright mask are
    folded in. The PSF model stamp (positive part) is centred on the nearest
    pixel, as the fit places it; stamp pixels beyond the image count as no
    data. A summed-area table finds the stamps that touch a no-data pixel at
    all, so only those are evaluated. Non-finite positions get 0 (never
    flagged); a position off the image gets 1.
    """
    from .psf_polynomial import evaluate_psf_at_batch_np
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u = np.zeros(x.shape, dtype=np.float64)
    nodata = np.asarray(nodata, dtype=bool)
    H, W = nodata.shape
    S = int(np.asarray(psf.coefs).shape[0])
    h = S // 2
    ok = np.isfinite(x) & np.isfinite(y)
    ix = np.zeros(x.shape, dtype=np.int64)
    iy = np.zeros(x.shape, dtype=np.int64)
    ix[ok] = np.rint(x[ok]).astype(np.int64)
    iy[ok] = np.rint(y[ok]).astype(np.int64)
    on = ok & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    u[ok & ~on] = 1.0
    if not np.any(on):
        return u
    padded = np.pad(nodata, h, constant_values=True)
    sat = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.int32)
    np.cumsum(padded, axis=0, dtype=np.int32, out=sat[1:, 1:])
    np.cumsum(sat[1:, 1:], axis=1, dtype=np.int32, out=sat[1:, 1:])
    # padded coords: stamp of a source at (iy, ix) = rows iy..iy+S-1, cols ix..ix+S-1
    r0, c0 = iy[on], ix[on]
    hits = (sat[r0 + S, c0 + S] - sat[r0, c0 + S]
            - sat[r0 + S, c0] + sat[r0, c0])
    idx = np.nonzero(on)[0][hits > 0]
    for s in range(0, idx.size, int(chunk)):
        c = idx[s:s + int(chunk)]
        st = np.clip(evaluate_psf_at_batch_np(psf, x[c], y[c]), 0.0,
                     None).astype(np.float64)
        tot = st.sum(axis=(1, 2))
        for k, j in enumerate(c):
            win = padded[iy[j]:iy[j] + S, ix[j]:ix[j] + S]
            u[j] = float((st[k] * win).sum() / tot[k]) if tot[k] > 0 else 1.0
    return u


def compose_band_flags(out: pd.DataFrame, bandnames, *,
                       psf_zone: dict | None = None,
                       saturated: dict | None = None,
                       partial: dict | None = None) -> dict:
    """R06: write ``FLAGS_{band}_fit`` (int16 bitmask, bits above) and
    ``n_bands_usable`` in-place.

    ``FLAG_NO_DATA`` follows the R05 rule (``FLUXERR_{band}_fit`` NaN);
    ``psf_zone`` maps band -> boolean array (``psf_lobe_zone``); bands missing
    from it get no PSF bit. ``partial`` maps band -> boolean array
    (``psf_uncovered_fraction > PARTIAL_COVERAGE_MAX``, R30): those cells get
    ``FLAG_PARTIAL_COVERAGE`` and keep their values. ``saturated`` maps band
    -> boolean array
    (saturated pixel within ``source_saturation_cut.radius_pix``, R11): those
    cells get ``FLAG_SATURATED`` and every per-band flux column blanked (the
    saturated core is masked, so what is left is a wing-only value).
    ``n_bands_usable`` counts the bands with ``FLAGS == 0`` and a finite flux;
    ``merge.compute_reliability_flag`` hard-fails a fitted source with none
    (``no_usable_band``), so one bad band no longer decides a source and an
    all-bad source is not called reliable.

    Returns ``{band: {"no_data": n, "psf_lobe": n, "saturated": n,
    "partial_coverage": n}}`` for the run log (all but ``no_data`` count
    cells that were otherwise measured).
    """
    psf_zone = psf_zone or {}
    saturated = saturated or {}
    partial = partial or {}
    n_rows = len(out)
    usable = np.zeros(n_rows, dtype=np.int16)
    counts = {}
    for bn in bandnames:
        fe_col = f"FLUXERR_{bn}_fit"
        if fe_col not in out.columns:
            continue
        fe = pd.to_numeric(out[fe_col], errors="coerce").to_numpy(
            dtype=np.float64)
        nodata = ~np.isfinite(fe)
        flags = np.where(nodata, FLAG_NO_DATA, 0).astype(np.int16)
        zone = psf_zone.get(bn)
        n_psf = 0
        if zone is not None:
            zone = np.asarray(zone, dtype=bool)
            flags |= np.where(zone, FLAG_PSF_LOBE, 0).astype(np.int16)
            n_psf = int(np.count_nonzero(zone & ~nodata))
        part = partial.get(bn)
        n_part = 0
        if part is not None:
            part = np.asarray(part, dtype=bool)
            flags |= np.where(part, FLAG_PARTIAL_COVERAGE, 0).astype(np.int16)
            n_part = int(np.count_nonzero(part & ~nodata))
        sat = saturated.get(bn)
        n_sat = 0
        if sat is not None:
            sat = np.asarray(sat, dtype=bool)
            flags |= np.where(sat, FLAG_SATURATED, 0).astype(np.int16)
            n_sat = int(np.count_nonzero(sat & ~nodata))
            if np.any(sat):
                for prefix in BAND_FLUX_COLUMN_PREFIXES:
                    col = f"{prefix}{bn}_fit"
                    if col in out.columns:
                        out.loc[sat, col] = np.nan
        out[f"FLAGS_{bn}_fit"] = flags
        fl_col = f"FLUX_{bn}_fit"
        fl = (pd.to_numeric(out[fl_col], errors="coerce").to_numpy(
                  dtype=np.float64)
              if fl_col in out.columns else np.full(n_rows, np.nan))
        usable += ((flags == 0) & np.isfinite(fl)).astype(np.int16)
        counts[bn] = {"no_data": int(np.count_nonzero(nodata)),
                      "psf_lobe": n_psf, "saturated": n_sat,
                      "partial_coverage": n_part}
    out["n_bands_usable"] = usable
    return counts
