"""Loading and validation of pipeline inputs.

This module owns the first major stage of :func:`tract7dt.pipeline.run_pipeline`:

1. Read the input source catalog and image list.
2. Per-frame parallel preparation (load FITS, scale to ``zp_ref``, build masks
   and inverse-variance maps, parse WCS).
3. Cross-frame validation (matching shapes, WCS alignment, FILTER coverage).
4. Inverse-variance-weighted ``white`` coadd (parallel over row chunks).
5. Optional crop, optional saturation flagging, optional overlay PNGs.
6. Persist a runtime ``wcs.fits`` snapshot (used later by ``merge``).

Returns a ``state`` dict consumed by every downstream stage.

Extracted from the original monolithic ``tract7dt.pipeline`` module in
Phase 1b. Behavior is intentionally unchanged.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval
from scipy.ndimage import distance_transform_edt, label as _nd_label
from scipy.spatial import cKDTree

from .utils import case_insensitive_col_map


logger = logging.getLogger("tract7dt.pipeline")


def _log(msg: str) -> None:
    text = str(msg).strip()
    if text.startswith("[WARN]"):
        logger.warning(text[len("[WARN]") :].strip())
    elif text.startswith("[CHECK]") and "FAILED" in text:
        logger.warning(text)
    else:
        logger.info(text)


def _read_image_list(list_path: Path, config_dir: Path) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Image list file not found: {list_path}")
    paths: list[Path] = []
    for line in list_path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = config_dir / p
        paths.append(p.resolve())
    return paths


def _frame_header_summary(fp: Path) -> dict:
    """The primary-header values the per-band frame choice needs."""
    hdr = fits.getheader(fp)

    def _num(key):
        try:
            v = float(hdr.get(key))
        except (TypeError, ValueError):
            return float("nan")
        return v if np.isfinite(v) else float("nan")

    return {
        "band": str(hdr.get("FILTER", "UNKNOWN")).strip(),
        "ul5_5": _num("UL5_5"),
        "zp_auto": _num("ZP_AUTO"),
        "skysig": _num("SKYSIG"),
        "telescop": str(hdr.get("TELESCOP", "")).strip(),
    }


def _select_frames_per_band(
    frames: list[tuple[Path, dict]],
) -> tuple[list[Path], pd.DataFrame]:
    """REGISTER R10: one frame per band, chosen deterministically.

    ``frames`` = ``(path, _frame_header_summary)`` in image-list order. A band
    listed more than once used to be decided by list order (``image_dict`` is
    keyed by band, so the LAST entry silently won). Now the deepest frame
    wins: header ``UL5_5`` (the 7DT pipeline's 5-sigma point-source depth,
    larger = deeper) when every frame of the band carries it, else the sky
    depth ``ZP_AUTO - 2.5 log10(SKYSIG)`` from keywords every frame must
    carry; an exact tie goes to the lexically first path, so the choice does
    not depend on list order.

    Returns the chosen frames, one per band, in the order of each band's first
    appearance (the band order the engine had before), and the table written
    to ``frames_used.csv`` (one row per listed frame).
    """
    bands: dict[str, list[int]] = {}
    for i, (_fp, h) in enumerate(frames):
        bands.setdefault(h["band"], []).append(i)
    rows: list[dict] = []
    chosen: list[Path] = []
    for band, idx in bands.items():
        ul5 = np.array([frames[i][1]["ul5_5"] for i in idx], dtype=float)
        if np.all(np.isfinite(ul5)):
            depth, key = ul5, "UL5_5"
        else:
            zp = np.array([frames[i][1]["zp_auto"] for i in idx], dtype=float)
            sk = np.array([frames[i][1]["skysig"] for i in idx], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                depth = zp - 2.5 * np.log10(sk)
            key = "ZP_AUTO-2.5log10(SKYSIG)"
        order = sorted(range(len(idx)), key=lambda k: (
            -(depth[k] if np.isfinite(depth[k]) else -np.inf),
            str(frames[idx[k]][0])))
        win = order[0]
        chosen.append(frames[idx[win]][0])
        for k, i in enumerate(idx):
            fp, h = frames[i]
            rows.append({
                "band": band, "file": str(fp), "used": k == win,
                "n_frames_band": len(idx), "depth_key": key,
                "depth_mag": float(depth[k]), "UL5_5": h["ul5_5"],
                "ZP_AUTO": h["zp_auto"], "SKYSIG": h["skysig"],
                "TELESCOP": h["telescop"],
            })
    return chosen, pd.DataFrame(rows)


def _wcs_close(w0: WCS, w1: WCS, tol: dict[str, float]) -> bool:
    a = w0.wcs
    b = w1.wcs

    def _allclose(x, y, key):
        t = float(tol.get(key, 0.0))
        return np.allclose(np.asarray(x), np.asarray(y), rtol=0.0, atol=t)

    if not _allclose(a.crval, b.crval, "crval"):
        return False
    if not _allclose(a.crpix, b.crpix, "crpix"):
        return False

    if a.has_cd() and b.has_cd():
        if not _allclose(a.cd, b.cd, "cd"):
            return False
    elif (not a.has_cd()) and (not b.has_cd()):
        if not _allclose(a.cdelt, b.cdelt, "cdelt"):
            return False
        if not _allclose(a.pc, b.pc, "cd"):
            return False
    else:
        return False

    if tuple(a.ctype) != tuple(b.ctype):
        return False
    return True


def _validate_headers(hdr, fp: Path) -> None:
    req = ["FILTER", "ZP_AUTO", "SKYSIG", "EGAIN"]
    for k in req:
        if k not in hdr:
            raise RuntimeError(f"Missing {k} in {fp}")
    if not str(hdr.get("FILTER", "")).strip():
        raise RuntimeError(f"Empty FILTER in {fp}")


def _zero_coverage_mask(img: np.ndarray) -> np.ndarray:
    """No-coverage mask for products that PAD outside their footprint with 0.0.

    Coadders differ in how they mark "no exposure here": NaN (already handled by
    the ``~isfinite`` test) or an exact 0.0 fill. The 0.0 convention is invisible
    to a finiteness test, so those pixels enter the fit as ordinary sky with a
    finite sigma -- sources there are measured against nothing and come out
    multiple magnitudes too faint while every quality flag stays clean
    (measured on 7DT coadds 2026-08-01: 11-24% of a tile's sources carry at
    least one such band, paired median vs DECaPS2 1.3-2.1 mag too faint on arm A, worst sub-population 3.8 mag, ``fracmasked`` 0.0000,
    ``fit_is_reliable`` True).

    This is HALF the rule. It returns border-connected exact zeros; the caller
    intersects that with :func:`_outside_data_envelope`, and the INTERSECTION is
    the whole no-coverage criterion (2026-08-01c) -- no threshold, no tuned
    constant anywhere. Do not use this function alone as "the mask".

    The half computed here is topological: exact-zero pixels CONNECTED TO THE
    FRAME BORDER. Interior zeros are left alone -- a legitimately-zero data
    pixel (or a zero-filled interior defect region that belongs in a real
    bad-pixel mask) is not our business here.
    """
    zero = img == 0.0
    if not zero.any():
        return zero
    # Connectivity notes (all three were live defects found by the 2026-08-01
    # record audit, fixed here):
    #  - label the union of zero AND non-finite pixels, then keep only the zero
    #    part. A product whose outer ring is NaN and whose next ring is 0 has no
    #    zero component touching the array border, and the naive test masked
    #    NOTHING while staying silent. Both conventions mean "no data", so they
    #    must percolate together.
    #  - 8-connectivity: a diagonal seam in the pad is one region, not two.
    pad = zero | ~np.isfinite(img)
    lab, n = _nd_label(pad, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(zero)
    border = np.unique(
        np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])
    )
    border = border[border > 0]
    if border.size == 0:
        # zeros exist but none reach the border -> interior, not padding
        return np.zeros_like(zero)
    return np.isin(lab, border) & zero


def _outside_data_envelope(img: np.ndarray) -> np.ndarray:
    """Pixels outside the ORTHOGONALLY CONVEX HULL of the non-zero data.

    A pixel is INSIDE only if it lies within both its row's and its column's
    non-zero span. This is the geometric half of the no-coverage rule and it
    carries no threshold.

    It must not be the axis-aligned bounding box: coadd footprints are ROTATED,
    and a rotated rectangle whose corners reach all four frame borders has a
    bounding box equal to the whole frame -- which is how the earlier bbox form
    reported "0.00% outside" against a 17.8% pad on T08285 m550/m575.
    """
    nz = img != 0.0
    anyr = np.any(nz, axis=1)
    anyc = np.any(nz, axis=0)
    if not (anyr.any() and anyc.any()):
        return np.ones_like(nz)
    h, w = nz.shape
    cidx = np.arange(w)[None, :]
    rmin = nz.argmax(axis=1)
    rmax = w - 1 - nz[:, ::-1].argmax(axis=1)
    row_in = (cidx >= rmin[:, None]) & (cidx <= rmax[:, None])
    row_in &= anyr[:, None]
    ridx = np.arange(h)[:, None]
    cmin = nz.argmax(axis=0)
    cmax = h - 1 - nz[::-1, :].argmax(axis=0)
    col_in = (ridx >= cmin[None, :]) & (ridx <= cmax[None, :])
    col_in &= anyc[None, :]
    return ~(row_in & col_in)


def _prepare_image_frame(fp: Path, zp_ref: float, saturation_divisor: float = 1.0,
                         mask_zero_coverage: bool = True) -> dict:
    """Load one FITS frame and build per-frame derived arrays.

    All large arrays are computed with in-place numpy ops where possible:
    a single ~213 MB float32 buffer per output (img_scaled,
    sigma_sky_scaled, sigma_total_scaled) instead of the previous
    pattern that allocated ~6 transient float32 buffers per frame
    (~1.5 GB per worker thread). With 4 parallel frame workers this
    drops the EPSF-stage RAM peak roughly in half and reduces the
    page-fault / allocator overhead of the per-frame numpy chain.
    """
    with fits.open(fp, memmap=True) as hdul:
        # np.array(..., copy=True) forces a fresh in-memory float32 copy
        # (vs np.asarray which can return a view of the soon-to-be-closed
        # mmap). Once we own the buffer we can mutate it in place.
        img = np.array(hdul[0].data, dtype=np.float32, copy=True)
        hdr = hdul[0].header

    _validate_headers(hdr, fp)

    filt = hdr.get("FILTER", "UNKNOWN").strip()
    zp = hdr.get("ZP_AUTO")
    if zp is None:
        raise RuntimeError(f"Missing ZP_AUTO in {fp}")

    skysig = hdr.get("SKYSIG")
    if skysig is None or (not np.isfinite(skysig)) or skysig <= 0:
        raise RuntimeError(f"Bad/Missing SKYSIG in {fp}")

    egain = hdr.get("EGAIN")
    if egain is None or (not np.isfinite(egain)) or egain <= 0:
        raise RuntimeError(f"Bad/Missing EGAIN (e-/ADU) in {fp}")

    warn_messages: list[str] = []
    if "SATURATE" not in hdr:
        warn_messages.append(f"[WARN] SATURATE missing in {fp}; using inf.")
        saturate = np.inf
    else:
        saturate = hdr.get("SATURATE", np.inf)
        try:
            if not np.isfinite(float(saturate)):
                warn_messages.append(f"[WARN] SATURATE invalid in {fp}; using inf.")
                saturate = np.inf
        except Exception:
            warn_messages.append(f"[WARN] SATURATE invalid in {fp}; using inf.")
            saturate = np.inf

    scale = float(10.0 ** (-0.4 * (float(zp) - zp_ref)))
    sigma_sky_adu = float(skysig)
    inv_egain = float(1.0 / float(egain))
    sigma_sky_var = float(sigma_sky_adu * sigma_sky_adu)
    scale_f32 = np.float32(scale)
    sky_scaled_value = np.float32(sigma_sky_adu * scale)

    # Compute bad / good masks once and reuse.
    bad = ~np.isfinite(img)
    if mask_zero_coverage:
        # A no-coverage pixel must satisfy BOTH conditions, and their
        # intersection is the whole rule -- there is no threshold anywhere:
        #   (1) it is an exact zero connected to the frame border  (value+topology)
        #   (2) it lies OUTSIDE the data's own envelope             (pure geometry)
        #
        # (2) is the percolation protection. On a product where exact zeros are
        # COMMON in real data (a non-negative-clipped sky), the border-connected
        # component leaks across the whole frame -- but almost none of it lies
        # outside the envelope, so the intersection keeps only the genuine sliver
        # instead of swallowing science pixels. This replaces an earlier
        # all-or-nothing guard that compared the two AREAS against a tuned
        # `2.0 x bound + 1%` slack: two constants, and an unsafe failure mode
        # (one bad comparison silently disabled masking for the entire band).
        # Measured 2026-08-01b on T08285 m400/m550/m575/m625/m700/m800 and on an
        # honest-L-pad control: the intersection is IDENTICAL to what the guarded
        # version masked, and on a clipped-sky product it masks the 0.42% truly
        # outside the envelope instead of refusing entirely.
        #
        # The envelope is the ORTHOGONALLY CONVEX HULL of the non-zero data (a
        # pixel is inside only if it lies within both its row's and its column's
        # non-zero span), NOT the axis-aligned bounding box: coadd footprints are
        # ROTATED, and a rotated rectangle whose corners reach all four borders
        # has a bounding box equal to the whole frame. With the bbox form,
        # `outside` collapsed to ~0 on exactly the worst-padded bands (T08285
        # m550/m575: 0.00% against a 17.8% pad, leaving 6.5% of the 500 px crop
        # unprotected; UDS m700 likewise -- and m700 POINT was the single
        # regression in the UDS deep-field A/B).
        nocov = _zero_coverage_mask(img)
        if nocov.any():
            n_border_zero = int(nocov.sum())
            nocov &= _outside_data_envelope(img)
            n_nocov = int(nocov.sum())
            if n_nocov:
                bad |= nocov
                msg = (f"[coverage] {fp.name}: {100.0 * n_nocov / nocov.size:.2f}% "
                       f"of pixels are outside the exposure footprint "
                       f"(border-connected no-data region) -> masked as no-coverage")
                if n_border_zero != n_nocov:
                    # Disclosure, not a threshold: border-connected zeros that sit
                    # INSIDE the data envelope are left alone, and we say how many.
                    msg += (f"; {100.0 * (n_border_zero - n_nocov) / nocov.size:.2f}% "
                            f"more is border-connected zero but lies INSIDE the data "
                            f"envelope and was left as data")
                warn_messages.append(msg)
            elif n_border_zero:
                warn_messages.append(
                    f"[coverage] {fp.name}: {100.0 * n_border_zero / nocov.size:.2f}% "
                    f"of pixels are border-connected exact zeros but ALL of them lie "
                    f"inside the data envelope -- zeros in this product look like "
                    f"DATA, not padding. Nothing masked.")
    good = ~bad

    # Saturation mask: reuse `good` instead of recomputing np.isfinite.
    satur_mask = np.zeros_like(bad)
    if np.isfinite(saturate):
        satur_thresh = float(saturate) / max(float(saturation_divisor), 1e-6)
        satur_mask = good & (img >= satur_thresh)

    # sigma_total_scaled = sqrt(sigma_sky_adu**2 + max(img_adu, 0)/egain) * scale
    # Compute using img (still in ADU) before we mutate it. One float32
    # buffer, all subsequent ops in place.
    sigma_total_scaled = np.empty_like(img)
    np.maximum(img, 0.0, out=sigma_total_scaled)
    sigma_total_scaled *= np.float32(inv_egain)
    sigma_total_scaled += np.float32(sigma_sky_var)
    np.sqrt(sigma_total_scaled, out=sigma_total_scaled)
    sigma_total_scaled *= scale_f32
    sigma_total_scaled[bad] = np.inf

    # Now we own `img`; mutate it in place to become `img_scaled`.
    img *= scale_f32
    img_scaled = img  # rename only; same buffer

    # sigma_sky_scaled: scalar fill, then patch bad pixels to inf.
    sigma_sky_scaled = np.full_like(img_scaled, sky_scaled_value)
    sigma_sky_scaled[bad] = np.inf

    wcs = WCS(hdr)
    return dict(
        filt=filt,
        shape=img_scaled.shape,
        wcs=wcs,
        img_scaled=img_scaled,
        sigma_total_scaled=sigma_total_scaled,
        sigma_sky_scaled=sigma_sky_scaled,
        bad=bad,
        satur_mask=satur_mask,
        hdr=hdr,
        scale=scale,
        fp=str(fp),
        warn_messages=warn_messages,
    )


def _accumulate_white_chunk(
    image_items: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    y0: int,
    y1: int,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    first_img = image_items[0][0]
    w = first_img.shape[1]
    h = y1 - y0
    white_num = np.zeros((h, w), dtype=np.float32)
    white_den = np.zeros((h, w), dtype=np.float32)
    white_mask = np.zeros((h, w), dtype=bool)

    for img_s, sig_sky, bad in image_items:
        img_c = np.asarray(img_s[y0:y1, :], dtype=np.float32)
        sig_c = np.asarray(sig_sky[y0:y1, :], dtype=np.float32)
        bad_c = np.asarray(bad[y0:y1, :], dtype=bool)

        # effarc2 K11 (2026-08-26): the same weight, formed WITHOUT boolean
        # fancy indexing. The masked form built two gather/scatter temporaries
        # per band per row-chunk; this is one fused elementwise pass. Exactly
        # equal: `good` pixels take the identical float32 expression, and the
        # rest are 0.0 either way (the `maximum(.., 1e-12)` guard means the
        # discarded branch never divides by zero, and a NaN sigma propagates
        # to a NaN that `where` drops). Measured on a real 22-band residual
        # stack: bit-identical, 2.3x faster.
        good = (~bad_c) & np.isfinite(sig_c) & (sig_c > 0)
        wt = np.where(good, 1.0 / np.maximum(sig_c * sig_c, np.float32(1e-12)),
                      np.float32(0.0)).astype(np.float32, copy=False)

        white_num += wt * img_c
        white_den += wt
        white_mask |= bad_c

    return y0, y1, white_num, white_den, white_mask


def _disk_offsets(radius_pix: float) -> list[tuple[int, int]]:
    r = max(0, int(np.ceil(float(radius_pix))))
    offs: list[tuple[int, int]] = []
    rr2 = float(radius_pix) * float(radius_pix)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if (dx * dx + dy * dy) <= rr2:
                offs.append((dy, dx))
    if not offs:
        offs.append((0, 0))
    return offs


def _sample_disk_mask(
    mask: np.ndarray,
    iy: np.ndarray,
    ix: np.ndarray,
    finite: np.ndarray,
    offsets: list[tuple[int, int]],
    *,
    chunk_size: int = 512,
) -> np.ndarray:
    """Sample whether any disk-offset pixel is true around each source."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    hit = np.zeros(len(ix), dtype=bool)
    if len(ix) == 0 or len(offsets) == 0:
        return hit

    n_finite = int(np.count_nonzero(finite))
    if (
        os.environ.get("TRACT7DT_BENCH_DISABLE_SAT_DISTANCE") != "1"
        and n_finite * len(offsets) > 500_000
    ):
        yy = iy[finite]
        xx = ix[finite]
        inb = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        local = np.zeros(n_finite, dtype=bool)
        if np.any(inb) and np.any(mask):
            # The distance to the nearest saturated pixel == the nearest-neighbour
            # Euclidean distance to the (sparse) set of True pixels, so a cKDTree
            # on the saturated-pixel coordinates returns the SAME distance as a
            # full-frame distance_transform_edt(~mask) -- bit-identical for the
            # `<= radius_pix` test -- but is far cheaper when saturation is sparse
            # (avoids transforming all ~5e7 pixels). radius_pix matches the disk
            # test dx*dx + dy*dy <= radius_pix**2 used to build `offsets`.
            radius_pix = float(max((dy * dy + dx * dx) ** 0.5 for dy, dx in offsets))
            sat_yx = np.argwhere(mask)  # (N_sat, 2) integer (y, x) of saturated px
            tree = cKDTree(sat_yx)
            q = np.column_stack([yy[inb], xx[inb]]).astype(float)
            d, _ = tree.query(q, k=1)
            local[inb] = d <= radius_pix
        hit[np.where(finite)[0]] = local
        return hit

    if (
        os.environ.get("TRACT7DT_BENCH_DISABLE_SAT_VECTOR") == "1"
        or n_finite * len(offsets) > 500_000
    ):
        for dy, dx in offsets:
            yy = iy + int(dy)
            xx = ix + int(dx)
            m = finite & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            if np.any(m):
                hit[m] |= mask[yy[m], xx[m]]
        return hit

    off_y = np.asarray([dy for dy, _dx in offsets], dtype=np.int64)[:, None]
    off_x = np.asarray([dx for _dy, dx in offsets], dtype=np.int64)[:, None]
    for start in range(0, len(ix), int(chunk_size)):
        stop = min(len(ix), start + int(chunk_size))
        yy = iy[start:stop][None, :] + off_y
        xx = ix[start:stop][None, :] + off_x
        valid = (
            finite[start:stop][None, :]
            & (yy >= 0) & (yy < H)
            & (xx >= 0) & (xx < W)
        )
        safe_y = np.clip(yy, 0, H - 1)
        safe_x = np.clip(xx, 0, W - 1)
        hit[start:stop] = np.any(valid & mask[safe_y, safe_x], axis=0)
    return hit


SATURATION_BAND_COLUMN_PREFIX = "_sat_"


def saturation_hits_per_band(
    image_dict: dict[str, dict],
    ix: np.ndarray,
    iy: np.ndarray,
    finite: np.ndarray,
    radius_pix: float,
) -> dict[str, np.ndarray]:
    """Per band: is a saturated pixel within ``radius_pix`` of the position?

    Bands without a ``satur_mask`` are left out (no saturation information).
    Shared by the catalog cut below and the injected-Gaia cut in ``zp.py``.
    """
    offs = _disk_offsets(radius_pix)
    hits: dict[str, np.ndarray] = {}
    for band, d in image_dict.items():
        sat = d.get("satur_mask", None)
        if sat is None:
            continue
        hits[band] = _sample_disk_mask(np.asarray(sat, dtype=bool), iy, ix,
                                       finite, offs)
    return hits


def _filter_sources_near_saturation(
    *,
    input_catalog: pd.DataFrame,
    image_dict: dict[str, dict],
    ra_col: str,
    dec_col: str,
    radius_pix: float,
    candidate_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Flag (not remove) sources near saturated pixels, per band (R11).

    Adds one boolean column per band, ``_sat_{band}`` (saturated pixel within
    ``radius_pix`` in that band; internal -- the GPU catalog folds it into
    ``FLAGS_{band}_fit`` bit 8 and blanks that band's values), and
    ``excluded_saturation`` = saturated in EVERY band, i.e. nothing left to
    measure. Up to dev7 a hit in ANY band excluded the source from the fit in
    all bands (dev7 anchors: 260 of 5,751 T08676 sources, 1,135 of 215,886
    T08285).
    """
    cat = input_catalog.copy()
    cat["excluded_saturation"] = False
    for band, d in image_dict.items():
        if d.get("satur_mask", None) is not None:
            cat[f"{SATURATION_BAND_COLUMN_PREFIX}{band}"] = False

    if len(cat) == 0:
        return cat

    wcs_white = next(iter(image_dict.values()))["wcs"]
    ra = pd.to_numeric(cat[ra_col], errors="coerce").to_numpy(dtype=float)
    dec = pd.to_numeric(cat[dec_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != finite.shape:
            raise ValueError(
                "candidate_mask shape mismatch for saturation filtering: "
                f"{candidate_mask.shape} != {finite.shape}"
            )
        finite &= candidate_mask
    if not np.any(finite):
        return cat

    x = np.full(len(cat), np.nan, dtype=float)
    y = np.full(len(cat), np.nan, dtype=float)
    xw, yw = wcs_white.all_world2pix(ra[finite], dec[finite], 0)
    x[finite] = np.asarray(xw, dtype=float)
    y[finite] = np.asarray(yw, dtype=float)
    finite &= np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return cat

    iy = np.rint(y).astype(int)
    ix = np.rint(x).astype(int)
    hits = saturation_hits_per_band(image_dict, ix, iy, finite, radius_pix)
    if not hits:
        return cat
    sat_any = np.zeros(len(cat), dtype=bool)
    sat_all = finite.copy()
    for band, hit in hits.items():
        cat[f"{SATURATION_BAND_COLUMN_PREFIX}{band}"] = hit
        sat_any |= hit
        sat_all &= hit
    cat["excluded_saturation"] = sat_all
    _log(f"Saturation (per band, radius_pix={float(radius_pix):.2f}): "
         f"{int(sat_any.sum())}/{len(cat)} sources near saturation in >=1 band; "
         f"{int(sat_all.sum())} in every band (excluded from the fit); per band: "
         + ", ".join(f"{b}={int(h.sum())}" for b, h in hits.items()))
    return cat


def load_inputs(cfg: dict) -> dict:
    inputs = cfg["inputs"]
    outputs = cfg["outputs"]
    checks = cfg["checks"]
    crop_cfg = cfg["crop"]
    overlay_cfg = cfg.get("overlay", {})
    sat_cut_cfg = cfg.get("source_saturation_cut", {})
    perf_cfg = cfg.get("performance", {})
    overlay_enabled = bool(overlay_cfg.get("enabled", True))
    overlay_downsample_full = max(
        1,
        int(overlay_cfg.get("downsample_full", 1)),
    )
    overlay_zoom_enabled = bool(overlay_cfg.get("zoom_enabled", False))
    overlay_zoom_box = overlay_cfg.get("zoom_box", None)
    t_load0 = time.perf_counter()
    t_prep = 0.0
    t_white = 0.0
    t_crop = 0.0
    t_sat = 0.0
    t_overlay = 0.0

    def _resolve_workers(raw: object, cap: int, *, label: str) -> int:
        if cap <= 0:
            return 1
        if isinstance(raw, str):
            if raw.strip().lower() == "auto":
                return cap
            try:
                parsed = int(raw)
            except Exception as e:
                raise ValueError(f"{label} must be 'auto' or an integer, got: {raw!r}") from e
            return max(1, min(cap, parsed))
        if isinstance(raw, (int, np.integer)):
            return max(1, min(cap, int(raw)))
        raise TypeError(f"{label} must be 'auto' or an integer, got: {type(raw).__name__}")

    cfg_path = cfg.get("config_path", None)
    if cfg_path is not None:
        _log(f"Config path: {cfg_path}")
    _log(f"Input catalog path: {inputs['input_catalog']}")
    _log(f"GaiaXP synphot path: {inputs.get('gaiaxp_synphot_csv', None)}")

    # Peek at just the header to resolve the ID column, so we can read the ID as
    # string in a SINGLE full pass (numeric-looking IDs like 00001 keep their
    # leading zeros) instead of reading the ~242 MB file twice. Bit-identical:
    # only the ID column's read dtype is forced; all other columns infer the same
    # whether or not ID is read as str.
    _header_cols = pd.read_csv(inputs["input_catalog"], nrows=0).columns
    _id_col0 = case_insensitive_col_map(_header_cols).get("id")
    _read_dtype = {_id_col0: str} if _id_col0 is not None else None
    input_catalog = pd.read_csv(inputs["input_catalog"], dtype=_read_dtype)
    col_map = case_insensitive_col_map(input_catalog.columns)
    ra_col = col_map.get("ra")
    dec_col = col_map.get("dec")
    type_col = col_map.get("type")
    if ra_col is None or dec_col is None:
        raise ValueError("input_catalog must have RA/DEC columns")
    if ra_col != "RA" or dec_col != "DEC":
        _log(f"[WARN] input_catalog uses {ra_col}/{dec_col} columns; interpreting as Right Ascension / Declination. Expected format is RA/DEC.")
    id_col = col_map.get("id")
    if id_col is not None:
        # ID was read as string above; normalize exactly as before.
        input_catalog[id_col] = input_catalog[id_col].fillna("").astype("string")
    input_catalog["excluded_crop"] = False
    input_catalog["excluded_saturation"] = False

    image_list = _read_image_list(inputs["image_list_file"], cfg["config_dir"])
    _log(f"7DT frames to stack: {len(image_list)} (list={inputs['image_list_file']})")
    for image_path in image_list:
        logger.debug("image_path=%s", image_path)
    if not image_list:
        raise FileNotFoundError(f"No image paths found in {inputs['image_list_file']}")

    zp_ref = float(cfg["image_scaling"]["zp_ref"])

    # R10: one frame per band (deepest by header), disclosed in
    # {work_dir}/frames_used.csv; frames not used are not loaded at all.
    all_frames = image_list
    image_list, frames_used = _select_frames_per_band(
        [(fp, _frame_header_summary(fp)) for fp in all_frames])
    work_dir = Path(cfg["outputs"]["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_used.to_csv(work_dir / "frames_used.csv", index=False)
    for band, grp in frames_used.groupby("band", sort=False):
        if len(grp) > 1:
            use = grp[grp["used"]].iloc[0]
            rest = grp[~grp["used"]]
            _log(f"[WARN] duplicate band {band}: {len(grp)} frames; using "
                 f"{Path(use['file']).name} ({use['depth_key']} "
                 f"{use['depth_mag']:.3f}); not used: "
                 + ", ".join(f"{Path(f).name} ({d:.3f})" for f, d in
                             zip(rest["file"], rest["depth_mag"])))
    if len(image_list) != len(all_frames):
        _log(f"Frames used: {len(image_list)} of {len(all_frames)} listed "
             f"(one per band; frames_used.csv)")

    image_dict: dict[str, dict] = {}
    ref_wcs = None
    ref_shape = None
    _log(f"Preparing image arrays/masks/WCS for {len(image_list)} frames...")
    t0 = time.perf_counter()
    # frame_prep is one task per FITS frame; the natural cap is the
    # number of frames itself (extra workers would just sit idle), but
    # the user's configured value is otherwise honored up to cpu_count
    # so 25-band runs scale automatically.
    cpu_n = int(os.cpu_count() or 1)
    prep_workers = _resolve_workers(
        perf_cfg.get("frame_prep_workers", 32),
        max(1, min(len(image_list), cpu_n)),
        label="performance.frame_prep_workers",
    )
    _log(f"Per-frame preparation workers: {prep_workers} (frames={len(image_list)}, cpu={cpu_n})")
    frame_results: dict[Path, dict] = {}

    with cf.ThreadPoolExecutor(max_workers=prep_workers) as ex:
        sat_div = float(sat_cut_cfg.get("saturation_divisor", 1.0))
        mask_zero_cov = bool(
            cfg.get("image_scaling", {}).get("mask_zero_coverage", True))
        futs = {ex.submit(_prepare_image_frame, fp, zp_ref, sat_div,
                          mask_zero_cov): fp for fp in image_list}
        for fut in cf.as_completed(futs):
            fp = futs[fut]
            frame_results[fp] = fut.result()

    for fp in image_list:
        fr = frame_results[fp]
        for msg in fr.get("warn_messages", []):
            _log(msg)

        filt = str(fr["filt"])
        shape = tuple(fr["shape"])
        wcs = fr["wcs"]

        if checks.get("require_same_shape", True):
            if ref_shape is None:
                ref_shape = shape
            elif shape != ref_shape:
                raise RuntimeError(f"Image shape mismatch: {fp} has {shape}, expected {ref_shape}")

        if checks.get("require_wcs_alignment", True):
            if ref_wcs is None:
                ref_wcs = wcs
            else:
                if not _wcs_close(ref_wcs, wcs, checks.get("wcs_tolerance", {})):
                    _log(f"[CHECK] WCS alignment FAILED for {fp}")
                    raise RuntimeError(f"WCS mismatch with reference: {fp}")

        image_dict[filt] = dict(
            img_scaled=fr["img_scaled"],
            sigma_total_scaled=fr["sigma_total_scaled"],
            sigma_sky_scaled=fr["sigma_sky_scaled"],
            bad=fr["bad"],
            satur_mask=fr["satur_mask"],
            hdr=fr["hdr"],
            wcs=wcs,
            scale=float(fr["scale"]),
            fp=fr["fp"],
        )
    t_prep = time.perf_counter() - t0
    _log(f"Finished per-frame preparation for {len(image_dict)} filters.")

    # Validate that input catalog band columns match image FILTERs, if provided.
    catalog_bands = {
        str(c[len("FLUX_") :]).strip()
        for c in input_catalog.columns
        if isinstance(c, str) and c.startswith("FLUX_") and len(c) > len("FLUX_")
    }
    if catalog_bands:
        image_bands = {str(b).strip() for b in image_dict.keys()}
        missing_in_images = sorted(catalog_bands - image_bands)
        missing_in_catalog = sorted(image_bands - catalog_bands)
        if missing_in_images or missing_in_catalog:
            raise RuntimeError(
                "Band mismatch between input catalog and image list. "
                f"catalog-only={missing_in_images} image-only={missing_in_catalog}"
            )
        _log("[CHECK] Input catalog bands match image FILTERs.")

    _log(f"Filters loaded: {list(image_dict.keys())}")
    if checks.get("require_wcs_alignment", True) and ref_wcs is not None:
        _log("[CHECK] WCS alignment PASSED for all images.")

    _log("Building white stack and white-noise map...")
    t0 = time.perf_counter()
    image_items = [
        (
            np.asarray(d["img_scaled"], dtype=np.float32),
            np.asarray(d["sigma_sky_scaled"], dtype=np.float32),
            np.asarray(d["bad"], dtype=bool),
        )
        for d in image_dict.values()
    ]
    h, w = image_items[0][0].shape
    white_num = np.zeros((h, w), dtype=np.float32)
    white_den = np.zeros((h, w), dtype=np.float32)
    white_mask = np.zeros((h, w), dtype=bool)

    # Adaptive row chunking: aim for roughly one chunk per requested
    # worker so that bumping `white_stack_workers` actually buys
    # parallelism. Floor of 64 rows / chunk to keep per-chunk overhead
    # amortized; cpu_count is the hard upper bound on useful workers.
    cpu_n = int(os.cpu_count() or 1)
    requested_white = perf_cfg.get("white_stack_workers", 32)
    if isinstance(requested_white, str) and requested_white.strip().lower() == "auto":
        target_chunks = cpu_n
    else:
        try:
            target_chunks = max(1, min(cpu_n, int(requested_white)))
        except Exception:
            target_chunks = cpu_n
    target_chunks = max(8, target_chunks)
    chunk_rows = max(64, h // target_chunks if target_chunks > 0 else h)
    row_ranges = [(y0, min(h, y0 + chunk_rows)) for y0 in range(0, h, chunk_rows)]
    white_workers = _resolve_workers(
        perf_cfg.get("white_stack_workers", 32),
        max(1, min(len(row_ranges), cpu_n)),
        label="performance.white_stack_workers",
    )
    _log(
        f"White stack workers: {white_workers} (row chunks={len(row_ranges)}, "
        f"chunk_rows={chunk_rows}, cpu={cpu_n})"
    )

    with cf.ThreadPoolExecutor(max_workers=white_workers) as ex:
        futs = [
            ex.submit(_accumulate_white_chunk, image_items, y0, y1)
            for y0, y1 in row_ranges
        ]
        for fut in cf.as_completed(futs):
            y0, y1, num_c, den_c, mask_c = fut.result()
            white_num[y0:y1, :] = num_c
            white_den[y0:y1, :] = den_c
            white_mask[y0:y1, :] = mask_c

    white = np.full_like(white_den, np.nan, dtype=np.float32)
    good = white_den > 0
    white[good] = (white_num[good] / white_den[good]).astype(np.float32)

    sigma_white = np.full_like(white_den, np.inf, dtype=np.float32)
    sigma_white[good] = np.sqrt(1.0 / white_den[good]).astype(np.float32)

    # Phase 1f cheap win: drop the intermediate accumulators so they're freed
    # before crop / overlay run. Each is float32 of the full white shape, so
    # for ~6800x10200 they're ~265 MB apiece (~800 MB freed total).
    del white_num, white_den, good
    t_white = time.perf_counter() - t0

    _log(f"White made: {white.shape}")

    if crop_cfg.get("enabled", True):
        t0_crop = time.perf_counter()
        CROP_MARGIN = int(crop_cfg.get("margin", 100))
        H0, W0 = white.shape
        x0 = int(CROP_MARGIN)
        x1 = int(W0 - CROP_MARGIN)
        y0 = int(CROP_MARGIN)
        y1 = int(H0 - CROP_MARGIN)

        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid crop region: white shape={(H0, W0)} margin={CROP_MARGIN}")

        _log(f"Crop box (pixel, half-open): {(x0, x1, y0, y1)} from white shape={(H0, W0)}")

        crop_out = outputs["cropped_images_dir"]
        crop_out.mkdir(parents=True, exist_ok=True)

        if crop_cfg.get("plot_pre_crop", True):
            _log("Generating pre-crop white diagnostic plot...")
            try:
                DS = int(crop_cfg.get("display_downsample", 4))
                white_show = np.asarray(white)[::DS, ::DS]
                m = np.isfinite(white_show)
                if np.any(m):
                    vmin, vmax = ZScaleInterval().get_limits(white_show[m])
                else:
                    vmin, vmax = None, None

                fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
                ax.imshow(
                    white_show,
                    origin="lower",
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    extent=(0, W0 - 1, 0, H0 - 1),
                    rasterized=True,
                )
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2.0))
                ax.set_title(f"White BEFORE crop (DS={DS}) with crop box")
                ax.set_xlim(0, W0 - 1)
                ax.set_ylim(0, H0 - 1)
                ax.set_xticks([])
                ax.set_yticks([])
                from .async_plot import submit_plot
                submit_plot(
                    fig,
                    crop_out / "white_before_crop.png",
                    dpi=int(cfg.get("plotting", {}).get("dpi", 150)),
                )
            except Exception as e:
                _log(f"[WARN] Could not plot pre-crop white: {e}")

        wcs0 = next(iter(image_dict.values()))["wcs"]
        _log("Applying crop to per-band arrays/WCS and flagging out-of-crop sources...")
        ra = pd.to_numeric(input_catalog[ra_col], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(input_catalog[dec_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(ra) & np.isfinite(dec)

        xpix, ypix = wcs0.all_world2pix(ra[ok], dec[ok], 0)
        xpix = np.asarray(xpix, dtype=float)
        ypix = np.asarray(ypix, dtype=float)

        in_crop = (xpix >= x0) & (xpix < x1) & (ypix >= y0) & (ypix < y1)

        keep = np.zeros(len(input_catalog), dtype=bool)
        keep[np.where(ok)[0][in_crop]] = True

        input_catalog["excluded_crop"] = ~keep
        n_crop_excluded = int((~keep).sum())
        _log(f"input_catalog: flagged {n_crop_excluded}/{len(input_catalog)} sources as excluded_crop")

        white = np.asarray(white)[y0:y1, x0:x1]
        sigma_white = np.asarray(sigma_white)[y0:y1, x0:x1]
        if white_mask is not None:
            try:
                white_mask = np.asarray(white_mask)[y0:y1, x0:x1]
            except Exception:
                pass

        for band, d in image_dict.items():
            d["img_scaled"] = np.asarray(d["img_scaled"])[y0:y1, x0:x1]
            d["sigma_total_scaled"] = np.asarray(d["sigma_total_scaled"])[y0:y1, x0:x1]
            d["sigma_sky_scaled"] = np.asarray(d["sigma_sky_scaled"])[y0:y1, x0:x1]
            d["bad"] = np.asarray(d["bad"])[y0:y1, x0:x1]
            if d.get("satur_mask", None) is not None:
                d["satur_mask"] = np.asarray(d["satur_mask"])[y0:y1, x0:x1]

            try:
                new_wcs = d["wcs"].slice((slice(y0, y1), slice(x0, x1)))
                d["wcs"] = new_wcs

                if "hdr" in d and d["hdr"] is not None:
                    new_hdr = d["hdr"].copy()
                    new_hdr.update(new_wcs.to_header(relax=True))
                    new_hdr["NAXIS1"] = int(x1 - x0)
                    new_hdr["NAXIS2"] = int(y1 - y0)
                    d["hdr"] = new_hdr
            except Exception as e:
                _log(f"[WARN] Could not slice WCS / update header for band={band}: {e}")

        _log(f"After crop: white shape={white.shape}")
        _log(f"After crop: first band shape={next(iter(image_dict.values()))['img_scaled'].shape}")

        if crop_cfg.get("plot_post_crop", True):
            try:
                _log("Generating post-crop white diagnostic plot...")
                ds_post = max(1, int(crop_cfg.get("post_crop_display_downsample", 1)))
                white_show = np.asarray(white)[::ds_post, ::ds_post]
                m = np.isfinite(white_show)
                if np.any(m):
                    vmin, vmax = ZScaleInterval().get_limits(white_show[m])
                else:
                    vmin, vmax = None, None
                fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
                ax.imshow(
                    white_show,
                    origin="lower",
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    extent=(0, white.shape[1] - 1, 0, white.shape[0] - 1),
                    rasterized=True,
                )
                ax.set_title(f"White AFTER crop (DS={ds_post})")
                ax.set_xticks([])
                ax.set_yticks([])
                from .async_plot import submit_plot
                submit_plot(
                    fig,
                    crop_out / "white_after_crop.png",
                    dpi=int(cfg.get("plotting", {}).get("dpi", 150)),
                )
            except Exception as e:
                _log(f"[WARN] Could not plot post-crop white: {e}")
        t_crop = time.perf_counter() - t0_crop

    if bool(sat_cut_cfg.get("enabled", False)):
        t0_sat = time.perf_counter()
        _log("Applying saturation-cut flagging on input catalog...")
        input_catalog = _filter_sources_near_saturation(
            input_catalog=input_catalog,
            image_dict=image_dict,
            ra_col=ra_col,
            dec_col=dec_col,
            radius_pix=float(sat_cut_cfg.get("radius_pix", 2.0)),
            candidate_mask=~input_catalog["excluded_crop"].fillna(False).to_numpy(dtype=bool),
        )
        t_sat = time.perf_counter() - t0_sat

    input_catalog["excluded_any"] = (
        input_catalog["excluded_crop"].fillna(False).astype(bool)
        | input_catalog["excluded_saturation"].fillna(False).astype(bool)
    )
    n_active = int((~input_catalog["excluded_any"]).sum())
    _log(f"Active (non-excluded) sources: {n_active}/{len(input_catalog)}")

    if overlay_enabled:
        t0_overlay = time.perf_counter()
        _log("Generating white+catalog overlay plot...")
        wcs_white = next(iter(image_dict.values()))["wcs"]
        H, W = white.shape
        fallback_model = str(cfg.get("patch_run", {}).get("gal_model", "dev")).upper()

        _is_active = ~input_catalog["excluded_any"].to_numpy(dtype=bool)
        _is_sat = input_catalog["excluded_saturation"].fillna(False).to_numpy(dtype=bool) & ~input_catalog["excluded_crop"].fillna(False).to_numpy(dtype=bool)
        n_crop_excluded = int(input_catalog["excluded_crop"].fillna(False).astype(bool).sum())
        n_sat_excluded = int(_is_sat.sum())

        active_cat = input_catalog.loc[_is_active].copy()
        ra = pd.to_numeric(active_cat[ra_col], errors="coerce").to_numpy(dtype=float)
        dec = pd.to_numeric(active_cat[dec_col], errors="coerce").to_numpy(dtype=float)
        if type_col is not None:
            type_series = active_cat[type_col].astype(str).str.strip().str.upper()
            type_series = type_series.replace({"": "UNKNOWN", "NAN": "UNKNOWN", "NONE": "UNKNOWN", "NULL": "UNKNOWN"})
            type_str = type_series.to_numpy()
        else:
            type_str = np.full(len(active_cat), "UNKNOWN", dtype=object)

        ok = np.isfinite(ra) & np.isfinite(dec)

        xpix, ypix = wcs_white.all_world2pix(ra[ok], dec[ok], 0)
        xpix = np.asarray(xpix, dtype=float)
        ypix = np.asarray(ypix, dtype=float)

        inb = (xpix >= 0) & (xpix < W) & (ypix >= 0) & (ypix < H)

        x_ok = xpix[inb]
        y_ok = ypix[inb]
        type_ok = type_str[ok][inb]

        is_star = type_ok == "STAR"
        is_gal = np.isin(type_ok, ["GAL", "EXP", "DEV", "SERSIC"])
        is_unknown = ~(is_star | is_gal)

        n_nan_or_oob = int(len(active_cat)) - int(ok.sum()) + int(ok.sum()) - int(inb.sum())

        x_sat = np.array([], dtype=float)
        y_sat = np.array([], dtype=float)
        sat_cat = input_catalog.loc[_is_sat]
        if len(sat_cat) > 0:
            _sat_ra = pd.to_numeric(sat_cat[ra_col], errors="coerce").to_numpy(dtype=float)
            _sat_dec = pd.to_numeric(sat_cat[dec_col], errors="coerce").to_numpy(dtype=float)
            _sat_ok = np.isfinite(_sat_ra) & np.isfinite(_sat_dec)
            if np.any(_sat_ok):
                _sx, _sy = wcs_white.all_world2pix(_sat_ra[_sat_ok], _sat_dec[_sat_ok], 0)
                _sx = np.asarray(_sx, dtype=float)
                _sy = np.asarray(_sy, dtype=float)
                _sinb = (_sx >= 0) & (_sx < W) & (_sy >= 0) & (_sy < H)
                x_sat = _sx[_sinb]
                y_sat = _sy[_sinb]

        DS_FULL = int(overlay_downsample_full)

        def _plot_white_with_overlay(*, white_arr: np.ndarray, extent: tuple[float, float, float, float], xlim, ylim, title: str):
            interval = ZScaleInterval()
            m = np.isfinite(white_arr)
            if np.any(m):
                vmin, vmax = interval.get_limits(white_arr[m])
            else:
                vmin, vmax = None, None

            fig = plt.figure(figsize=(10, 7), constrained_layout=True)
            ax = fig.add_subplot(111, projection=wcs_white)
            ax.imshow(
                white_arr,
                origin="lower",
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                extent=extent,
                rasterized=True,
            )

            ax.coords[0].set_axislabel("RA", fontsize=12)
            ax.coords[1].set_axislabel("Dec", fontsize=12)

            secax_x = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
            secax_y = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
            secax_x.set_xlabel("x [pix]", fontsize=12)
            secax_y.set_ylabel("y [pix]", fontsize=12)

            if np.any(is_star):
                ax.scatter(
                    x_ok[is_star],
                    y_ok[is_star],
                    s=20,
                    facecolors="none",
                    edgecolors="cyan",
                    linewidths=0.8,
                    label=f"STAR (N={is_star.sum()})",
                    transform=ax.get_transform("pixel"),
                )
            if np.any(is_gal):
                ax.scatter(
                    x_ok[is_gal],
                    y_ok[is_gal],
                    s=20,
                    facecolors="none",
                    edgecolors="magenta",
                    linewidths=0.8,
                    label=f"GAL (N={is_gal.sum()})",
                    transform=ax.get_transform("pixel"),
                )
            if np.any(is_unknown):
                ax.scatter(
                    x_ok[is_unknown],
                    y_ok[is_unknown],
                    s=24,
                    facecolors="none",
                    edgecolors="yellow",
                    linewidths=0.9,
                    marker="s",
                    label=f"UNKNOWN (N={is_unknown.sum()})\n(Fallback to {fallback_model})",
                    transform=ax.get_transform("pixel"),
                )

            if len(x_sat) > 0:
                ax.scatter(
                    x_sat,
                    y_sat,
                    s=60,
                    color="red",
                    linewidths=1.2,
                    marker="x",
                    zorder=5,
                    label=f"Sat-excl (N={n_sat_excluded})",
                    transform=ax.get_transform("pixel"),
                )
            elif n_sat_excluded > 0:
                ax.plot([], [], "rx", markersize=6, label=f"Sat-excl (N={n_sat_excluded})")

            if n_crop_excluded > 0:
                ax.plot([], [], "o", mfc="none", mec="gray", markersize=5,
                        label=f"Crop-excl (N={n_crop_excluded})")

            if n_nan_or_oob > 0:
                ax.plot([], [], "d", mfc="none", mec="gray", markersize=4,
                        label=f"NaN/OOB (N={n_nan_or_oob})")

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_title(title)
            ax.legend(loc="upper right", frameon=True, fontsize=12)
            return fig

        white_native = np.asarray(white, dtype=np.float32)
        white_full = white_native[::DS_FULL, ::DS_FULL]
        fig = _plot_white_with_overlay(
            white_arr=white_full,
            extent=(0, W - 1, 0, H - 1),
            xlim=(0, W - 1),
            ylim=(0, H - 1),
            title=None,
        )
        overlay_out = outputs["overlay_dir"]
        overlay_out.mkdir(parents=True, exist_ok=True)
        from .async_plot import submit_plot
        submit_plot(
            fig,
            overlay_out / "white_overlay.png",
            dpi=int(cfg.get("plotting", {}).get("dpi", 150)),
        )

        if overlay_zoom_enabled:
            try:
                if not isinstance(overlay_zoom_box, (list, tuple)) or len(overlay_zoom_box) != 4:
                    raise ValueError("overlay.zoom_box must be [x0, x1, y0, y1]")
                zx0, zx1, zy0, zy1 = [int(v) for v in overlay_zoom_box]
                if zx1 <= zx0 or zy1 <= zy0:
                    raise ValueError(
                        f"overlay.zoom_box must satisfy x1>x0 and y1>y0, got {[zx0, zx1, zy0, zy1]}"
                    )

                zoom_w = int(zx1 - zx0)
                zoom_h = int(zy1 - zy0)
                zoom_arr = np.full((zoom_h, zoom_w), np.nan, dtype=np.float32)

                src_x0 = max(0, zx0)
                src_x1 = min(W, zx1)
                src_y0 = max(0, zy0)
                src_y1 = min(H, zy1)
                if src_x1 > src_x0 and src_y1 > src_y0:
                    dst_x0 = int(src_x0 - zx0)
                    dst_x1 = int(dst_x0 + (src_x1 - src_x0))
                    dst_y0 = int(src_y0 - zy0)
                    dst_y1 = int(dst_y0 + (src_y1 - src_y0))
                    zoom_arr[dst_y0:dst_y1, dst_x0:dst_x1] = white_native[src_y0:src_y1, src_x0:src_x1]

                fig_zoom = _plot_white_with_overlay(
                    white_arr=zoom_arr,
                    extent=(zx0, zx1 - 1, zy0, zy1 - 1),
                    xlim=(zx0, zx1 - 1),
                    ylim=(zy0, zy1 - 1),
                    title=f"Overlay zoom [{zx0}:{zx1}, {zy0}:{zy1}]",
                )
                from .async_plot import submit_plot
                submit_plot(
                    fig_zoom,
                    overlay_out / "white_overlay_zoom.png",
                    dpi=int(cfg.get("plotting", {}).get("dpi", 150)),
                )
            except Exception as e:
                _log(f"[WARN] Could not generate white overlay zoom plot: {e}")
        t_overlay = time.perf_counter() - t0_overlay

    wcs_fits = None
    if image_dict:
        first_band = next(iter(image_dict.values()))
        try:
            # x_pix_white(_fit) is in the current white-frame coordinates
            # (post-crop when crop is enabled). Persist a matching WCS snapshot
            # so merge computes RA/DEC in the same pixel frame.
            merge_wcs_path = Path(outputs["work_dir"]) / "wcs.fits"
            merge_wcs_path.parent.mkdir(parents=True, exist_ok=True)

            wcs_obj = first_band.get("wcs", None)
            if wcs_obj is None:
                raise RuntimeError("missing first-band WCS object")
            hdr = wcs_obj.to_header(relax=True)
            # Keep this file tiny: only header fidelity matters for WCS transforms.
            fits.PrimaryHDU(data=np.zeros((1, 1), dtype=np.uint8), header=hdr).writeto(
                merge_wcs_path,
                overwrite=True,
            )
            wcs_fits = str(merge_wcs_path)
            _log(f"Saved WCS snapshot: {wcs_fits}")
        except Exception as e:
            raise RuntimeError(
                "Failed to save runtime WCS snapshot (wcs.fits). "
                "This snapshot is required to compute RA_fit/DEC_fit in the working pixel frame."
            ) from e

    state = dict(
        input_catalog=input_catalog,
        image_dict=image_dict,
        white=white,
        sigma_white=sigma_white,
        wcs_fits=wcs_fits,
    )
    t_total = time.perf_counter() - t_load0
    _log(
        "load_inputs timing [s]: prep=%.2f white=%.2f crop=%.2f sat=%.2f overlay=%.2f total=%.2f"
        % (t_prep, t_white, t_crop, t_sat, t_overlay, t_total)
    )
    return state
