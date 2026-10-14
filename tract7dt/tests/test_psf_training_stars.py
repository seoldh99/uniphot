"""PSF training-star rule (M2 unit a.3): who may train the PSF, and design 6 test P5(a).

Design: development design note design_v2.md (not distributed) 1.5 (v2.4; designer answers D8, D9-4) and
6 (P5); plan impl_plan.md 1.3 a.3.  Expected values come from the design rules and the synthetic
truth (tests/synth, engine-free), never from the module under test:

- candidates are SEP detections paired one-to-one with an input row that has a Gaia identity
  (column GAIA_SOURCE_ID) and passes the Gaia quality (RUWE <= 1.4, NON_SINGLE_STAR = 0; input
  columns); a list without any Gaia identity falls back to TYPE == STAR rows;
- hard exclusions are properties of the data only: a saturated pixel or a bad pixel in the
  star's cutout, a peak at or above the linear limit, a brighter detection's peak inside the
  star's 55 px box; S/N >= 20 defines the pool; there is no ellipticity cut and no separation
  rule between candidates;
- P5(a): planted (truth) stars, blank spots and saturated stars never train the PSF, and the
  model peak equals the truth within 0.2 %; without the saturation handling (positive control)
  the peak is off by more than 0.5 %.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import synth
from tract7dt.fit.grid_gpu import psf_from_stars as pfs
from tract7dt.fit.grid_gpu import psf_model as pm



def _ident(ra, dec):
    return np.asarray(ra, dtype=float), np.asarray(dec, dtype=float)


# ---------------------------------------------------------------------------
# Which input rows may train the PSF
# ---------------------------------------------------------------------------


def _catalog():
    return pd.DataFrame({
        "ID": ["g0", "g1", "g2", "g3", "t4", "b5", "g6", "g7", "g8"],
        "RA": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, np.nan],
        "DEC": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        "TYPE": ["STAR", "STAR", "STAR", "STAR", "STAR", "STAR", "STAR", "GAL", "STAR"],
        "GAIA_SOURCE_ID": ["123", "124", "125", "126", "", None, "127", "128", "129"],
        "RUWE": [1.0, 1.5, 1.2, np.nan, np.nan, np.nan, 1.4, 1.1, 1.0],
        "NON_SINGLE_STAR": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    })


def test_training_rows_are_gaia_identity_rows_that_pass_the_quality():
    rows = pfs.psf_training_rows(_catalog(), lambda ra, dec: (2.0 * np.asarray(ra), 3.0 * np.asarray(dec)),
                                 max_ruwe=1.4)
    assert rows.origin == "gaia"
    np.testing.assert_array_equal(rows.index, np.arange(8))           # the row without RA is left out
    np.testing.assert_allclose(rows.x, 2.0 * np.arange(10.0, 81.0, 10.0))
    np.testing.assert_allclose(rows.y, 3.0 * np.arange(1.0, 9.0))
    # g0 ok; g1 RUWE > 1.4; g2 non-single; g3 RUWE unknown; t4/b5 no identity; g6 RUWE = 1.4 ok;
    # g7 a Gaia identity whatever its TYPE
    np.testing.assert_array_equal(rows.trainable, [True, False, False, False, False, False, True, True])


def test_non_single_star_column_is_optional():
    cat = _catalog().drop(columns=["NON_SINGLE_STAR"])
    rows = pfs.psf_training_rows(cat, _ident, max_ruwe=1.4)
    np.testing.assert_array_equal(rows.trainable, [True, False, True, False, False, False, True, True])


@pytest.mark.parametrize("how", ["no_column", "all_empty"])
def test_list_without_gaia_identity_falls_back_to_star_rows(how):
    cat = _catalog()
    if how == "no_column":
        cat = cat.drop(columns=["GAIA_SOURCE_ID"])
    else:
        cat["GAIA_SOURCE_ID"] = ""
    rows = pfs.psf_training_rows(cat, _ident, max_ruwe=1.4)
    assert rows.origin == "star"
    np.testing.assert_array_equal(rows.trainable, [True] * 7 + [False])
    assert "GAIA_SOURCE_ID" in rows.note and "planted" in rows.note


def test_gaia_identity_without_ruwe_stops_the_run():
    with pytest.raises(ValueError, match="RUWE"):
        pfs.psf_training_rows(_catalog().drop(columns=["RUWE"]), _ident, max_ruwe=1.4)


def test_pairing_is_one_to_one_within_the_radius():
    det_x = np.array([10.0, 50.0, 90.0, 91.0, 130.0])
    det_y = np.array([10.0, 50.0, 90.0, 90.0, 130.0])
    row_x = np.array([10.5, 50.5, 49.0, 90.5, 133.0, 200.0])
    row_y = np.array([10.0, 50.0, 50.5, 90.0, 130.0, 200.0])
    got = pfs.pair_detections_with_rows(det_x, det_y, row_x, row_y, radius=2.0)
    # unique pair; two rows near one detection; two detections near one row (both);
    # nearest row too far
    np.testing.assert_array_equal(got, [0, -1, -1, -1, -1])


# ---------------------------------------------------------------------------
# Hard exclusions and the cuts that are gone
# ---------------------------------------------------------------------------


def _objs(spec):
    dt = np.dtype([("x", "f8"), ("y", "f8"), ("a", "f8"), ("b", "f8"), ("flux", "f8"),
                   ("fluxerr", "f8"), ("peak", "f8"), ("xpeak", "i8"), ("ypeak", "i8")])
    o = np.zeros(len(spec), dtype=dt)
    for k, (x, y, flux, snr, peak, a, b) in enumerate(spec):
        o[k] = (x, y, a, b, flux, flux / snr, peak, int(np.rint(x)), int(np.rint(y)))
    return o


#        name  x      y      flux   S/N   peak    a    b    row: (has Gaia id, RUWE)
_STARS = [("A", 100.3, 100.6, 1e4, 200., 5e3, 1.5, 1.5, ("1", 1.0)),
          ("B", 300.2, 100.1, 1e4, 200., 5e3, 1.5, 1.5, ("2", 1.0)),     # saturated pixel in cutout
          ("C", 100.4, 300.2, 2e5, 900., 9e4, 1.5, 1.5, ("3", 1.0)),     # peak >= linear limit
          ("D", 300.5, 300.5, 1e4, 200., 5e3, 1.5, 1.5, ("4", 1.0)),     # bad pixel in cutout
          ("E", 200.0, 200.0, 1e4, 200., 5e3, 1.5, 1.5, ("5", 1.0)),     # brighter peak in box (F)
          ("F", 220.0, 190.0, 3e4, 400., 1.5e4, 1.5, 1.5, None),         # detection without a row
          ("G", 200.0, 60.0, 1e4, 200., 5e3, 1.5, 1.5, ("7", 1.0)),      # G and H: 20 px apart,
          ("H", 220.0, 60.0, 1e4, 200., 5e3, 1.5, 1.5, ("8", 1.0)),      #  equally bright: both stay
          ("I", 60.0, 200.0, 1e3, 15., 5e2, 1.5, 1.5, ("9", 1.0)),       # S/N < 20
          ("J", 60.0, 360.0, 1e4, 200., 5e3, 1.5, 1.5, ("", np.nan)),    # planted: no Gaia identity
          ("K", 360.0, 200.0, 1e4, 200., 5e3, 1.5, 1.5, ("11", 2.0)),    # RUWE 2
          ("L", 20.0, 200.0, 1e4, 200., 5e3, 1.5, 1.5, ("12", 1.0)),     # cutout leaves the image
          ("M", 150.0, 250.0, 1e4, 200., 5e3, 3.0, 1.0, ("13", 1.0))]    # elongated: stays


def _exclusion_case():
    objs = _objs([s[1:8] for s in _STARS])
    with_row = [s for s in _STARS if s[8] is not None]
    cat = pd.DataFrame({"RA": [s[1] + 0.3 for s in with_row], "DEC": [s[2] - 0.2 for s in with_row],
                        "TYPE": "STAR", "GAIA_SOURCE_ID": [s[8][0] for s in with_row],
                        "RUWE": [s[8][1] for s in with_row]})
    shape = (420, 420)
    sat = np.zeros(shape, dtype=bool)
    sat[100, 325] = True                       # 25 px from B: inside its 61 px cutout
    bad = np.zeros(shape, dtype=bool)
    bad[271, 329] = True                       # (29, -29) px from D: inside its cutout
    rows = pfs.psf_training_rows(cat, _ident, max_ruwe=1.4)
    return objs, rows, shape, sat, bad


def test_hard_exclusions_and_no_separation_or_ellipticity_cut():
    objs, rows, shape, sat, bad = _exclusion_case()
    sel, counts, paired = pfs.select_psf_stars(
        objs, image_shape=shape, fwhm_mode=4.0, min_snr=20.0, edge_pad_pix=10 + 30,
        cutout_size=61, psf_size=55, bad_mask=bad, sat_mask=sat, linear_limit=8e4, rows=rows)
    names = [_STARS[i][0] for i in sel]
    assert sorted(names) == ["A", "G", "H", "M"]
    assert counts == {"total": 13, "after_edge": 12, "after_identity": 9, "after_snr": 8,
                      "after_saturation": 7, "after_linear_limit": 6, "after_bad_pixels": 5,
                      "after_brighter_neighbour": 4}
    with_row = [s[0] for s in _STARS if s[8] is not None]
    assert [with_row[rows.index[p]] for p in paired] == names


def test_each_hard_exclusion_is_needed():
    """Positive controls: switching each data rule off lets its star back in."""
    objs, rows, shape, sat, bad = _exclusion_case()
    kw = dict(image_shape=shape, fwhm_mode=4.0, min_snr=20.0, edge_pad_pix=40, cutout_size=61,
              psf_size=55, rows=rows)
    base = dict(bad_mask=bad, sat_mask=sat, linear_limit=8e4)
    for change, back in [({"sat_mask": np.zeros(shape, bool)}, "B"),
                         ({"linear_limit": np.inf}, "C"),
                         ({"bad_mask": np.zeros(shape, bool)}, "D")]:
        sel, _, _ = pfs.select_psf_stars(objs, **kw, **{**base, **change})
        assert sorted(_STARS[i][0] for i in sel) == sorted(["A", "G", "H", "M", back])


def test_retired_selection_keys_are_gone_from_the_defaults(config):
    fs = config.DEFAULT_CONFIG["compute"]["gpu"]["psf"]["from_stars"]
    assert "max_ellipticity" not in fs and "min_sep" not in fs
    assert fs["min_snr"] == 20.0 and fs["max_stars_per_band"] == 5000


# ---------------------------------------------------------------------------
# P5(a): planted stars, blank spots and saturated stars never train the PSF
# ---------------------------------------------------------------------------

_P5 = dict(shape=(2000, 2600), n_gaia=500, n_planted=200, n_blank=100, n_sat=25, seed=5)


def _p5_scene():
    rng = np.random.default_rng(_P5["seed"])
    h, w = _P5["shape"]
    psf = synth.PSFField((h, w), fwhm=4.5)                                    # the truth
    planted_psf = synth.PSFField((h, w), profile="moffat", beta=2.0, fwhm=6.0)  # truth-B-like shape
    n_g, n_p, n_b, n_s = _P5["n_gaia"], _P5["n_planted"], _P5["n_blank"], _P5["n_sat"]
    x, y = synth.draw_positions((h, w), n_g + n_p + n_b, rng, margin=45.0)
    st_g = psf.stamp(1000.0, 1000.0).data
    st_p = planted_psf.stamp(1000.0, 1000.0).data
    f_g = synth.sky_limited_flux(synth.draw_snr(n_g, 50.0, 2000.0, rng), st_g)
    f_p = synth.sky_limited_flux(synth.draw_snr(n_p, 50.0, 2000.0, rng), st_p)
    model, _ = synth.render((h, w), x[n_s:n_g], y[n_s:n_g], f_g[n_s:], psf)
    model_p, _ = synth.render((h, w), x[n_g:n_g + n_p], y[n_g:n_g + n_p], f_p, planted_psf)
    model += model_p
    saturate = 1.5 * float(model.max())                 # nothing else comes near the limit
    k = 10.0 ** rng.uniform(np.log10(2.0), np.log10(8.0), n_s)   # 5 % of the candidates: peak 2-8 x
    f_g[:n_s] = k * saturate / st_g.max()
    model_s, _ = synth.render((h, w), x[:n_s], y[:n_s], f_g[:n_s], psf)
    model += model_s
    data = synth.realize(model, rng)
    clipped = data >= saturate
    data = np.minimum(data, saturate)
    sigma = np.sqrt(1.0 + np.maximum(data, 0.0) / 50.0)
    kind = np.array(["gaia"] * n_g + ["planted"] * n_p + ["blank"] * n_b)
    cat = pd.DataFrame({"RA": x, "DEC": y, "TYPE": "STAR",
                        "GAIA_SOURCE_ID": [str(10 ** 12 + i) if kind[i] == "gaia" else "" for i in range(len(x))],
                        "RUWE": np.where(kind == "gaia", 1.0, np.nan), "NON_SINGLE_STAR": 0})
    is_sat = np.zeros(len(x), dtype=bool)
    is_sat[:n_s] = True
    flux = np.concatenate([f_g, f_p, np.zeros(n_b)])
    return dict(psf=psf, data=data, sigma=sigma, clipped=clipped, saturate=saturate, cat=cat,
                kind=kind, is_sat=is_sat, model=model, flux=flux)


def _train(sc, saturation_handling: bool):
    data, sigma = sc["data"], sc["sigma"]
    h, w = data.shape
    bad = np.zeros(data.shape, dtype=bool)
    divisor = 1.3                       # DEFAULT_CONFIG source_saturation_cut.saturation_divisor
    sat = data >= sc["saturate"] / divisor if saturation_handling else np.zeros(data.shape, bool)
    limit = sc["saturate"] if saturation_handling else np.inf     # header SATURATE fallback
    objs = pfs.detect_sep_objects(data, sigma, bad, thresh_sigma=5.0, minarea=25,
                                  deblend_nthresh=32, deblend_cont=0.005)
    x, y, fwhm, _, snr = pfs._objs_with_quality_metrics(objs)
    fm = pfs.compute_fwhmrange_psfex(fwhm[snr > 5.0], sample_variability=0.2)[0]
    rows = pfs.psf_training_rows(sc["cat"], _ident, max_ruwe=1.4)
    sel, counts, paired = pfs.select_psf_stars(
        objs, image_shape=(h, w), fwhm_mode=fm, min_snr=20.0, edge_pad_pix=10 + 30,
        cutout_size=61, psf_size=55, bad_mask=bad, sat_mask=sat, linear_limit=limit, rows=rows)
    cat_rows = rows.index[paired]
    stamps, fx, fy, keep = pfs.extract_centered_stamps(
        data, sigma, bad, x[sel], y[sel], cutout_size=61, psf_size=55,
        star_local_bkg_subtract=False, centering_method="gw_centroid", fwhm_pix=fm)
    ix, iy = np.rint(fx).astype(int), np.rint(fy).astype(int)
    box_flux = np.array([data[j - 27:j + 28, i - 27:i + 28].sum() for i, j in zip(ix, iy)])
    # iteration-0 noise as the a.2 wiring will call it (prereg_a.3_r5d.md 1.4: list rows, paired rows, saturation)
    a, info = pfs.psf0_training_noise(data, objs, sel[keep], fx, fy, box_flux, stamps,
                                      np.ones_like(data), 50.0, bad, fm, rows_x=rows.x, rows_y=rows.y,
                                      star_row=paired[keep], sat_mask=sat, half=27)   # sky sigma 1, gain 50
    model = pm.build_psf_model(stamps, fx, fy, a, (h, w))
    gx, gy = np.meshgrid(np.linspace(300.0, w - 300.0, 4), np.linspace(300.0, h - 300.0, 3))
    peaks = model.render(gx.ravel(), gy.ravel())[:, 27, 27]
    truth = sc["psf"].stamp(1000.0, 1000.0).data[27, 27]
    clipped_in_cutout = np.array([sc["clipped"][j - 30:j + 31, i - 30:i + 31].any()
                                  for i, j in zip(np.rint(x[sel]).astype(int), np.rint(y[sel]).astype(int))])
    return dict(peak_err=float(np.max(np.abs(peaks / truth - 1.0))), counts=counts,
                n_planted=int((sc["kind"][cat_rows] == "planted").sum()),
                n_blank=int((sc["kind"][cat_rows] == "blank").sum()),
                n_sat=int(sc["is_sat"][cat_rows].sum()),
                n_clipped=int(clipped_in_cutout.sum()), n_train=int(len(fx)),
                profile=info["profile"], n_isolated=info["n_isolated"], kappa=info["kappa"])


@pytest.fixture(scope="module")
def p5_scene():
    return _p5_scene()


@pytest.fixture(scope="module")
def p5_trained(p5_scene):
    return _train(p5_scene, saturation_handling=True)


def test_p5a_no_planted_blank_or_saturated_star_trains(p5_trained):
    r = p5_trained
    assert r["n_planted"] == 0 and r["n_blank"] == 0 and r["n_sat"] == 0 and r["n_clipped"] == 0, r
    assert r["n_train"] >= 200, r


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "P5(a) model peak FAILED in development (PSF(0) of this scene: 1.43 % in round 1, 1.36 % in round 3, "
    "against <= 0.2 %). Kept as a recorded failure by the author's decision; the final peak is to be "
    "judged on the PSF after its next development step, on new seeds. Not a pass"))
def test_p5a_model_peak_equals_the_truth(p5_trained):
    assert p5_trained["peak_err"] <= 2e-3, p5_trained


def test_p5a_positive_control_without_saturation_handling(p5_scene, p5_trained):
    r = _train(p5_scene, saturation_handling=False)
    assert r["n_sat"] > 0, r                    # the rule is what keeps them out
    assert r["peak_err"] > 5e-3, r
    assert r["peak_err"] > 3.0 * p5_trained["peak_err"], (r, p5_trained)


# ---------------------------------------------------------------------------
# The PSF stage hands every band its saturation mask, linear limit and training rows
# ---------------------------------------------------------------------------


def _stage_case(tmp_path, n_bands=1, workers=1, scale=2.0):
    import copy

    from astropy.io import fits
    from astropy.wcs import WCS

    from tract7dt.config import DEFAULT_CONFIG

    rng = np.random.default_rng(11)
    h, w = 520, 640
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [w / 2, h / 2]
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.cdelt = [-0.505 / 3600, 0.505 / 3600]
    psf = synth.PSFField((h, w), fwhm=4.5)
    gx, gy = np.meshgrid(np.linspace(60, w - 61, 8), np.linspace(60, h - 61, 6))
    x = gx.ravel() + rng.uniform(-3, 3, gx.size)
    y = gy.ravel() + rng.uniform(-3, 3, gy.size)
    flux = synth.sky_limited_flux(synth.draw_snr(len(x), 100.0, 1000.0, rng), psf.stamp(300.0, 300.0).data)
    planted = np.array([5, 17, 30])                       # stars without a Gaia identity
    sat_star = 22                                         # a Gaia star clipped at SATURATE
    flux[sat_star] = 6.0 * flux.max()
    model, _ = synth.render((h, w), x, y, flux, psf)
    saturate_img = 0.5 * float(model[int(round(y[sat_star])), int(round(x[sat_star]))])
    other = np.ones(len(x), dtype=bool)
    other[sat_star] = False
    assert float(synth.render((h, w), x[other], y[other], flux[other], psf)[0].max()) < saturate_img / 1.5
    data = np.minimum(synth.realize(model, rng), saturate_img)
    ra, dec = wcs.all_pix2world(x, y, 0)
    cat = pd.DataFrame({"ID": [f"s{i}" for i in range(len(x))], "RA": ra, "DEC": dec, "TYPE": "STAR",
                        "GAIA_SOURCE_ID": ["" if i in planted else str(4 * 10 ** 15 + i) for i in range(len(x))],
                        "RUWE": [np.nan if i in planted else 1.0 for i in range(len(x))]})
    image_dict = {}
    for b in range(n_bands):
        hdr = fits.Header()
        hdr["SATURATE"] = saturate_img / scale           # raw ADU; image units = raw x scale
        img = data.astype(np.float32)
        image_dict[f"m{500 + 25 * b}"] = dict(
            img_scaled=img, sigma_total_scaled=np.sqrt(1.0 + np.maximum(img, 0) / 50.0).astype(np.float32),
            bad=np.zeros((h, w), bool), satur_mask=img >= saturate_img / 1.3, hdr=hdr, wcs=wcs,
            scale=scale, fp=str(tmp_path / f"frame{b}.fits"))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["outputs"]["work_dir"] = str(tmp_path)
    cfg["compute"]["gpu"]["psf"]["from_stars"]["max_workers"] = workers
    cfg["compute"]["gpu"]["psf"]["from_stars"]["save_diagnostics"] = True
    state = {"image_dict": image_dict, "input_catalog": cat}
    return cfg, state, dict(x=x, y=y, planted=planted, sat_star=sat_star, saturate=saturate_img)


def test_stage_passes_the_band_saturation_mask_linear_limit_and_rows(tmp_path, monkeypatch):
    from tract7dt.fit.grid_gpu import psf_pipeline_stage as stage

    cfg, state, truth = _stage_case(tmp_path)
    seen = {}

    def spy(band, image, sigma, bad_mask, *, sat_mask, cfg, rows, linear_limit):
        seen.update(band=band, sat_mask=sat_mask, rows=rows, linear_limit=linear_limit)
        raise RuntimeError("spy stops the build")

    monkeypatch.setattr(stage, "build_psf_polynomial_from_stars_for_band", spy)
    stage.build_psf_polynomial_from_stars_stage(cfg, state)
    bd = state["image_dict"]["m500"]
    np.testing.assert_array_equal(seen["sat_mask"], bd["satur_mask"])
    assert seen["sat_mask"].any()
    assert seen["linear_limit"] == pytest.approx(truth["saturate"], rel=1e-12)   # SATURATE x scale
    rows = seen["rows"]
    assert rows.origin == "gaia"
    np.testing.assert_allclose(rows.x, truth["x"], atol=1e-6)
    np.testing.assert_allclose(rows.y, truth["y"], atol=1e-6)
    want = np.ones(len(truth["x"]), dtype=bool)
    want[truth["planted"]] = False
    np.testing.assert_array_equal(rows.trainable, want)


@pytest.mark.parametrize("workers", [1, 2])
def test_stage_trains_without_planted_or_saturated_stars(tmp_path, workers):
    from tract7dt.fit.grid_gpu import psf_pipeline_stage as stage

    cfg, state, truth = _stage_case(tmp_path, n_bands=2, workers=workers)
    try:
        stage.build_psf_polynomial_from_stars_stage(cfg, state)
    finally:
        stage.shutdown_psf_diagnostics()
    summary = pd.read_csv(tmp_path / "psf_polynomial" / "summary.csv")
    assert list(summary["status"]) == ["ok", "ok"], summary.to_dict("records")
    for band in ("m500", "m525"):
        used = pd.read_csv(tmp_path / "psf_polynomial" / band / "stars.csv")
        assert len(used) >= 20 and set(used["star_origin"]) == {"gaia"}
        d = np.hypot(used["star_x"].to_numpy()[:, None] - truth["x"][None, :],
                     used["star_y"].to_numpy()[:, None] - truth["y"][None, :])
        nearest = d.argmin(axis=1)
        assert d.min(axis=1).max() < 1.0
        assert not set(nearest) & set(truth["planted"].tolist()), band
        assert truth["sat_star"] not in set(nearest), band


def test_no_input_rows_gives_no_training_star():
    objs, _, shape, sat, bad = _exclusion_case()
    rows = pfs.psf_training_rows(pd.DataFrame({"RA": [np.nan], "DEC": [1.0], "TYPE": ["STAR"]}),
                                 _ident, max_ruwe=1.4)
    assert len(rows.x) == 0
    sel, counts, paired = pfs.select_psf_stars(
        objs, image_shape=shape, fwhm_mode=4.0, min_snr=20.0, edge_pad_pix=40, cutout_size=61,
        psf_size=55, bad_mask=bad, sat_mask=sat, linear_limit=8e4, rows=rows)
    assert len(sel) == 0 and len(paired) == 0 and counts["after_identity"] == 0
