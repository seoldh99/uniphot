"""Which white-stack detections are the light of listed sources (bright-object mask association, uniphot 0.1.1).

Why: the bright-object mask covers detections on the white stack that are not the light of a source in the input
list. In 0.1.0 a detection counted as listed only when its isophotal centroid lay within a fixed 3 px of a list
position, on a detection made without removing the sky. The public example lost 99 of its 751 rows that way: on 92
the sky pedestal widened the detection and pulled its centroid 3-8.5 px off the source, on 7 the list position is
3.3-10.6 px from the star's light. Each was masked by its own light and not measured.

The rule tested here: a detection's position is its matched-filter peak pixel; it is the light of a listed source
when an active row lies within r of that peak, where for each population of rows (rows of the user's list, Gaia
rows added by the engine as ZP calibrators)
    r = min(pos_max_shift_pix, max(FWHM, k * sigma)),  k = sqrt(-2 ln 1e-3) = 3.72,
FWHM is the run's PSF FWHM, and sigma is the list-image scatter measured in the same run (mutual nearest row-peak
pairs with detection S/N >= 20, 1.4826 x the median absolute offset per axis about 0, combined as the rms of the two
axes; not used when there are fewer than 30 pairs). pos_max_shift_pix is the distance the fit may move a source.

Expected values are independent of the implementation: they follow from the construction of each scene (unit noise,
a circular Gaussian PSF of sigma 1.6 px = FWHM 3.77 px, or a Moffat beta = 3 of the same FWHM, flux set by the PSF
optimal S/N, list offsets drawn by fixed seeds). They are the expected values of the design for this change, written
before the implementation; the sub-pixel phases of the source grids are those of the designer's feasibility scenes.
Every scene goes through the engine's own detection (``detect_bright_sources``) and association + mask
(``finalize_bright_mask``) with the default configuration, except that unmatched detections are masked rather than
subtracted (``subtract_unmatched: false``), so that "a source's own light is masked" is visible in the mask.

Observables: the mask at a source's true position (True = its own light was masked), ``affected_by_bright_mask`` of
the rows, the per-detection classification ``state["bright_mask_matched"]`` and the run record
``bright_mask/association.json`` (FWHM, per-population scatter, pair count and radius).
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from tract7dt.bright_mask import detect_bright_sources, finalize_bright_mask
from tract7dt.config import DEFAULT_CONFIG

N = 1200
SIG = 1.6
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0)) * SIG       # 3.77 px
CAP = 5.0                                           # pos_max_shift_pix used in every scene
K = np.sqrt(-2.0 * np.log(1e-3))                    # 3.72
BAND = "m600"
# The FWHM the engine reads from the run's PSF goes through a radial profile binned in 0.5 px annuli;
# one bin in diameter is the tolerance on the recorded FWHM.
FWHM_TOL = 0.5


# ---------------------------------------------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------------------------------------------
def flux_for_snr(snr: float) -> float:
    """Total flux of a Gaussian (sigma SIG) point source with PSF-optimal S/N ``snr`` in unit noise."""
    return float(snr) * np.sqrt(4.0 * np.pi * SIG ** 2)


def add_gauss(img, x0, y0, flux):
    r = int(6 * SIG) + 2
    ys = slice(max(int(y0) - r, 0), min(int(y0) + r + 1, img.shape[0]))
    xs = slice(max(int(x0) - r, 0), min(int(x0) + r + 1, img.shape[1]))
    yy, xx = np.mgrid[ys, xs]
    img[ys, xs] += flux / (2 * np.pi * SIG ** 2) * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * SIG ** 2))


def moffat_alpha(beta=3.0):
    return FWHM / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))


def add_moffat(img, x0, y0, flux, beta=3.0, r=60):
    a = moffat_alpha(beta)
    ys = slice(max(int(y0) - r, 0), min(int(y0) + r + 1, img.shape[0]))
    xs = slice(max(int(x0) - r, 0), min(int(x0) + r + 1, img.shape[1]))
    yy, xx = np.mgrid[ys, xs]
    img[ys, xs] += flux * (beta - 1) / (np.pi * a ** 2) * (1 + ((xx - x0) ** 2 + (yy - y0) ** 2) / a ** 2) ** (-beta)


def psf_stamp(kind="gauss", size=55):
    img = np.zeros((size, size))
    c = (size - 1) / 2.0
    if kind == "gauss":
        add_gauss(img, c, c, 1.0)
    else:
        add_moffat(img, c, c, 1.0, r=size)
    return img / img.sum()


def grid_positions(n_side, spacing, start, n):
    """Source grid with the sub-pixel phases of the designer's feasibility scenes (0-0.3 px)."""
    pos = [(start + i * spacing + 0.37 * ((i * 7 + j * 3) % 5) / 5,
            start + j * spacing + 0.29 * ((i * 3 + j * 5) % 5) / 5)
           for j in range(n_side) for i in range(n_side)]
    return np.array(pos[:n], float)


def scene1(pedestal, noise_seed=11):
    """49 isolated point sources, S/N 20-300, list offsets 0-2 px, sky pedestal + 0.5 sigma gradient along x."""
    k = np.arange(49)
    snrs = np.geomspace(20.0, 300.0, 49)
    rad = 2.0 * k / 48.0
    ang = k * 2.39996
    offs = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    img = np.random.default_rng(noise_seed).normal(0.0, 1.0, (N, N))
    img = img + pedestal + 0.5 * np.arange(N)[None, :] / (N - 1)
    pos = grid_positions(7, 150, 150, 49)
    for (x0, y0), s in zip(pos, snrs):
        add_gauss(img, x0, y0, flux_for_snr(s))
    return img.astype(np.float32), pos, pos + offs


# ---------------------------------------------------------------------------------------------------------------
# engine call
# ---------------------------------------------------------------------------------------------------------------
def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [N / 2 + 0.5, N / 2 + 0.5]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    return w


def run_engine(tmp_path, img, rows_xy, injected=None, sigma=None, psf="gauss", tag="run"):
    """The engine's detection + association + mask on one scene. ``rows_xy``: list positions (0-based pixels)."""
    rows_xy = np.asarray(rows_xy, float)
    injected = np.zeros(len(rows_xy), bool) if injected is None else np.asarray(injected, bool)
    sigma = np.ones_like(img, dtype=np.float32) if sigma is None else np.asarray(sigma, np.float32)
    root = tmp_path / tag
    wcs = _wcs()
    ra, dec = wcs.all_pix2world(rows_xy[:, 0], rows_xy[:, 1], 0)
    cat = pd.DataFrame({"ID": [f"r{i}" for i in range(len(rows_xy))], "RA": ra, "DEC": dec,
                        "TYPE": "STAR", "excluded_any": False, "_is_injected_gaia": injected})
    state = {"white": img, "sigma_white": sigma, "image_dict": {BAND: {"wcs": wcs}}, "input_catalog": cat}
    if psf is not None:
        d = root / "psf" / BAND
        d.mkdir(parents=True)
        st = psf_stamp(psf)
        np.savez(d / "psf_polynomial.npz", coefs=st[:, :, None].astype(np.float32), x_origin=0.0, x_scale=1.0,
                 y_origin=0.0, y_scale=1.0, H_psf=st.shape[0], W_psf=st.shape[1])
        state["psf_polynomial_dir"] = str(root / "psf")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["outputs"]["work_dir"] = str(root / "work")
    cfg["bright_mask"]["subtract_unmatched"] = False
    cfg["bright_mask"]["save_overlay"] = False
    cfg["compute"]["gpu"]["positions"]["pos_max_shift_pix"] = CAP
    objs = detect_bright_sources(cfg, state)
    mask = finalize_bright_mask(cfg, state, objs)
    return {"objs": objs, "mask": mask, "state": state, "cat": state["input_catalog"],
            "record_path": root / "work" / "bright_mask" / "association.json"}


def own_detection(objs, pos):
    """Index of the detection whose centroid is nearest each true source position, and that distance."""
    d, i = cKDTree(np.column_stack([objs["x"], objs["y"]])).query(np.asarray(pos, float))
    return i, d


def masked_at(mask, pos):
    pos = np.asarray(pos, float)
    return mask[np.round(pos[:, 1]).astype(int), np.round(pos[:, 0]).astype(int)]


def record(res):
    with open(res["record_path"]) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------------------------------------------
# 1. classification does not depend on the sky pedestal
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("pedestal", [0.0, 1.6, 5.0, -0.5])
def test_isolated_sources_are_their_own_light_whatever_the_sky(tmp_path, pedestal):
    img, pos, rows = scene1(pedestal)
    res = run_engine(tmp_path, img, rows)
    objs = res["objs"]
    i_own, d_own = own_detection(objs, pos)
    assert int(np.sum(d_own <= 1.5)) == 49, f"{int(np.sum(d_own > 1.5))} sources have no detection within 1.5 px"
    d_det, _ = cKDTree(pos).query(np.column_stack([objs["x"], objs["y"]]))
    assert int(np.sum(d_det > 1.5)) <= 2, f"{int(np.sum(d_det > 1.5))} detections are not a source"
    assert int(objs["npix"].max()) < 2000, f"largest detection {int(objs['npix'].max())} px"
    assert not res["cat"]["affected_by_bright_mask"].any(), \
        f"{int(res['cat']['affected_by_bright_mask'].sum())} rows inside the bright-object mask"
    assert not masked_at(res["mask"], pos).any()
    matched = np.asarray(res["state"]["bright_mask_matched"], bool)
    assert matched.shape == (len(objs),)
    assert matched[i_own].all(), f"{int((~matched[i_own]).sum())} sources' own detections unmatched"


# ---------------------------------------------------------------------------------------------------------------
# 2. a list position farther than the fit can move is not the source's light
# ---------------------------------------------------------------------------------------------------------------
def test_light_beyond_the_fit_reach_is_unmatched(tmp_path):
    img, pos, rows = scene1(1.6)
    pos8 = np.array([(75.3 + 150 * i, 75.2) for i in range(8)])
    ang = np.arange(8) * np.pi / 4
    rows8 = pos8 + 8.0 * np.column_stack([np.cos(ang), np.sin(ang)])
    for x0, y0 in pos8:
        add_gauss(img, x0, y0, flux_for_snr(50.0))
    res = run_engine(tmp_path, img, np.vstack([rows, rows8]))
    assert masked_at(res["mask"], pos8).all(), "a source 8 px from its list position was not masked"
    assert not masked_at(res["mask"], pos).any()
    i49, d49 = own_detection(res["objs"], pos)
    i8, d8 = own_detection(res["objs"], pos8)
    assert (d49 <= 1.5).all() and (d8 <= 1.5).all()
    matched = np.asarray(res["state"]["bright_mask_matched"], bool)
    assert not matched[i8].any()
    assert matched[i49].all()


# ---------------------------------------------------------------------------------------------------------------
# 3. the radius follows the measured list-image scatter, between the FWHM and the fit's reach
# ---------------------------------------------------------------------------------------------------------------
def _scene3(sigma_list, offset_seed):
    snr = np.geomspace(30.0, 100.0, 300)
    off = np.random.default_rng(offset_seed).normal(0.0, sigma_list, (300, 2))
    img = np.random.default_rng(5).normal(0.0, 1.0, (N, N))
    pos = grid_positions(18, 60, 90, 300)
    for (x0, y0), s in zip(pos, snr):
        add_gauss(img, x0, y0, flux_for_snr(s))
    return img.astype(np.float32), pos, pos + off, off


def test_poor_list_widens_the_radius_up_to_the_fit_reach(tmp_path):
    img, pos, rows, off = _scene3(1.5, 3)
    res = run_engine(tmp_path, img, rows)
    beyond = int(np.sum(np.hypot(off[:, 0], off[:, 1]) > CAP))
    n_masked = int(masked_at(res["mask"], pos).sum())
    assert abs(n_masked - beyond) <= 1, f"{n_masked} sources masked; {beyond} list offsets exceed {CAP} px"
    rec = record(res)
    pop = rec["populations"]["listed"]
    assert pop["scatter_measured"] is True
    assert 1.35 <= pop["sigma_pix"] <= 1.75, pop
    assert pop["radius_pix"] == pytest.approx(CAP, abs=1e-9), pop
    i_own, d_own = own_detection(res["objs"], pos)
    assert (d_own <= 1.5).all()
    matched = np.asarray(res["state"]["bright_mask_matched"], bool)
    assert abs(int((~matched[i_own]).sum()) - beyond) <= 1


def test_good_list_uses_the_psf_fwhm(tmp_path):
    img, pos, rows, _ = _scene3(0.3, 4)
    res = run_engine(tmp_path, img, rows)
    assert not masked_at(res["mask"], pos).any()
    rec = record(res)
    assert rec["fwhm_pix"] == pytest.approx(FWHM, abs=FWHM_TOL), rec["fwhm_pix"]
    pop = rec["populations"]["listed"]
    assert pop["scatter_measured"] is True
    assert 0.25 <= pop["sigma_pix"] <= 0.5, pop
    assert pop["radius_pix"] == pytest.approx(rec["fwhm_pix"], abs=1e-9), pop
    i_own, _ = own_detection(res["objs"], pos)
    assert np.asarray(res["state"]["bright_mask_matched"], bool)[i_own].all()


# ---------------------------------------------------------------------------------------------------------------
# 4. a bright unlisted source whose detection area holds a faint listed row is still unlisted (and the mirror)
# ---------------------------------------------------------------------------------------------------------------
BRIGHT = (600.2, 600.4)
FAINT = (612.2, 600.1)


def _scene4():
    img = np.random.default_rng(21).normal(0.0, 1.0, (N, N))
    add_moffat(img, *BRIGHT, flux_for_snr(3000.0))
    add_moffat(img, *FAINT, flux_for_snr(10.0))
    others = np.array([(150 + 150 * (i % 7), 150 + 150 * (i // 7)) for i in range(49)
                       if (i % 7, i // 7) != (3, 3)][:40], float)
    for x0, y0 in others:
        add_moffat(img, x0, y0, flux_for_snr(60.0))
    return img.astype(np.float32), others


def _faint_row_inside_bright_segment(img):
    """Precondition: the faint row's pixel belongs to the bright source's detection area (unit noise, no sky)."""
    import sep
    from tract7dt.sep_limits import apply_sep_limits
    apply_sep_limits(sep)
    p = DEFAULT_CONFIG["bright_mask"]
    objs, seg = sep.extract(img, thresh=p["thresh_sigma"], err=np.ones_like(img), minarea=p["minarea"],
                            deblend_nthresh=p["deblend_nthresh"], deblend_cont=p["deblend_cont"],
                            clean=p["clean"], clean_param=p["clean_param"], segmentation_map=True)
    ib = int(np.argmin(np.hypot(objs["x"] - BRIGHT[0], objs["y"] - BRIGHT[1])))
    return int(seg[int(round(FAINT[1])), int(round(FAINT[0]))]) == ib + 1


def test_bright_unlisted_source_is_not_claimed_by_a_faint_row_in_its_area(tmp_path):
    img, others = _scene4()
    assert _faint_row_inside_bright_segment(img), "scene precondition"
    rows = np.vstack([others + [0.5, 0.0], [FAINT]])
    res = run_engine(tmp_path, img, rows, psf="moffat")
    ib, db = own_detection(res["objs"], [BRIGHT])
    assert db[0] <= 1.5
    _, d_faint = own_detection(res["objs"], [FAINT])
    assert d_faint[0] > 3.0, "the faint source has its own detection: the scene does not test the rule"
    assert masked_at(res["mask"], [BRIGHT]).all(), "the bright unlisted source was not masked"
    assert not np.asarray(res["state"]["bright_mask_matched"], bool)[ib[0]]


def test_bright_listed_source_is_matched(tmp_path):
    img, others = _scene4()
    rows = np.vstack([others + [0.5, 0.0], [BRIGHT]])
    res = run_engine(tmp_path, img, rows, psf="moffat")
    ib, db = own_detection(res["objs"], [BRIGHT])
    assert db[0] <= 1.5
    assert not masked_at(res["mask"], [BRIGHT]).any()
    assert np.asarray(res["state"]["bright_mask_matched"], bool)[ib[0]]


# ---------------------------------------------------------------------------------------------------------------
# 5. close pairs of listed stars: merged or split, no unmatched detection
# ---------------------------------------------------------------------------------------------------------------
def test_close_listed_pairs_leave_no_unmatched_detection(tmp_path):
    img = np.random.default_rng(31).normal(0.0, 1.0, (N, N))
    pairs = []
    for k, (sep_px, ratio) in enumerate([(5, 1), (5, 3), (6, 1), (6, 3), (7, 1), (7, 3)]):
        a = (200.0 + 150 * k, 600.3)
        b = (a[0] + sep_px, a[1] + 0.4)
        add_gauss(img, *a, flux_for_snr(50.0))
        add_gauss(img, *b, flux_for_snr(50.0) * ratio)
        pairs.append((a, b))
    rows = np.array([p for ab in pairs for p in ab])
    res = run_engine(tmp_path, img.astype(np.float32), rows)
    objs = res["objs"]
    matched = np.asarray(res["state"]["bright_mask_matched"], bool)
    yy, xx = np.mgrid[0:N, 0:N]
    for a, b in pairs:
        cx, cy = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
        near = np.hypot(objs["x"] - cx, objs["y"] - cy) < 10
        assert near.any(), (a, b)
        assert matched[near].all(), f"pair at {a}: {int((~matched[near]).sum())} unmatched detections"
        win = (slice(int(cy) - 12, int(cy) + 13), slice(int(cx) - 12, int(cx) + 13))
        assert not (res["mask"][win] & (np.hypot(xx[win] - cx, yy[win] - cy) < 10)).any()


# ---------------------------------------------------------------------------------------------------------------
# 6. the scatter is measured per population (user list / added Gaia rows)
# ---------------------------------------------------------------------------------------------------------------
def test_scatter_is_measured_per_population(tmp_path):
    rng = np.random.default_rng(41)
    snr = np.geomspace(30.0, 150.0, 400)
    off = np.r_[rng.normal(0.0, 1.2, (200, 2)), rng.normal(0.0, 0.15, (200, 2))]
    inj = np.r_[np.zeros(200, bool), np.ones(200, bool)]
    perm = rng.permutation(400)
    snr, off, inj = snr[perm], off[perm], inj[perm]
    img = np.random.default_rng(42).normal(0.0, 1.0, (N, N))
    pos = grid_positions(20, 55, 75, 400)
    for (x0, y0), s in zip(pos, snr):
        add_gauss(img, x0, y0, flux_for_snr(s))
    res = run_engine(tmp_path, img.astype(np.float32), pos + off, injected=inj)
    assert not masked_at(res["mask"], pos).any(), f"{int(masked_at(res['mask'], pos).sum())} sources masked"
    rec = record(res)
    lst, gai = rec["populations"]["listed"], rec["populations"]["injected_gaia"]
    assert lst["n_rows"] == 200 and gai["n_rows"] == 200
    assert 1.1 <= lst["sigma_pix"] <= 1.45, lst
    assert 0.2 <= gai["sigma_pix"] <= 0.45, gai
    assert 4.1 <= lst["radius_pix"] <= 5.0, lst
    assert gai["radius_pix"] == pytest.approx(rec["fwhm_pix"], abs=1e-9), gai
    i_own, _ = own_detection(res["objs"], pos)
    matched = np.asarray(res["state"]["bright_mask_matched"], bool)
    assert matched[i_own][~inj].all() and matched[i_own][inj].all()


# ---------------------------------------------------------------------------------------------------------------
# 7. fewer than 30 pairs: no scatter term, radius = FWHM; without a PSF the radius is the fit's reach
# ---------------------------------------------------------------------------------------------------------------
def _scene7():
    img = np.random.default_rng(61).normal(0.0, 1.0, (N, N))
    pos = grid_positions(4, 200, 250, 10)
    for x0, y0 in pos:
        add_gauss(img, x0, y0, flux_for_snr(60.0))
    return img.astype(np.float32), pos


def test_too_few_pairs_use_the_fwhm(tmp_path):
    img, pos = _scene7()
    res = run_engine(tmp_path, img, pos + [0.5, 0.0])
    rec = record(res)
    pop = rec["populations"]["listed"]
    assert pop["scatter_measured"] is False and pop["sigma_pix"] is None, pop
    assert pop["radius_pix"] == pytest.approx(rec["fwhm_pix"], abs=1e-9), pop
    assert rec["fwhm_pix"] == pytest.approx(FWHM, abs=FWHM_TOL)
    i_own, _ = own_detection(res["objs"], pos)
    assert np.asarray(res["state"]["bright_mask_matched"], bool)[i_own].all()


def test_without_a_psf_profile_the_radius_is_the_fit_reach(tmp_path):
    img, pos = _scene7()
    res = run_engine(tmp_path, img, pos + [0.5, 0.0], psf=None)
    rec = record(res)
    assert rec["fwhm_pix"] is None
    assert rec["populations"]["listed"]["radius_pix"] == pytest.approx(CAP, abs=1e-9)


# ---------------------------------------------------------------------------------------------------------------
# 8. invariance: shifting the scene, scaling image and noise together
# ---------------------------------------------------------------------------------------------------------------
def _classify(res, pos):
    i_own, d_own = own_detection(res["objs"], pos)
    assert (d_own <= 1.5).all()
    matched = np.asarray(res["state"]["bright_mask_matched"], bool)
    return (matched[i_own], masked_at(res["mask"], pos), res["cat"]["affected_by_bright_mask"].to_numpy(bool))


def test_classification_is_shift_and_scale_invariant(tmp_path):
    img, pos, rows = scene1(1.6)
    base = run_engine(tmp_path, img, rows, tag="base")
    dx, dy = 37, 11
    img_s = np.full_like(img, np.nan)
    img_s[dy:, dx:] = img[:-dy, :-dx]
    sig_s = np.ones_like(img)
    sig_s[np.isnan(img_s)] = np.nan
    shifted = run_engine(tmp_path, img_s, rows + [dx, dy], sigma=sig_s, tag="shift")
    scaled = run_engine(tmp_path, img * np.float32(7.3), rows, sigma=np.full_like(img, 7.3), tag="scale")
    c0 = _classify(base, pos)
    for name, res, p in (("shift", shifted, pos + [dx, dy]), ("scale", scaled, pos)):
        c1 = _classify(res, p)
        for a, b in zip(c0, c1):
            assert np.array_equal(a, b), name
    r0, r2 = record(base), record(scaled)
    assert r2["populations"]["listed"]["radius_pix"] == pytest.approx(r0["populations"]["listed"]["radius_pix"],
                                                                     rel=1e-6)
