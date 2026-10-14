"""Band-level reason flags (REGISTER R06) and the PSF-lobe zone (R07).

R07. Where the band's PSF model stamp dips below -0.2% of its peak (mostly the
extrapolated zone outside the PSF training stars, where the spatial polynomial
runs away), forced PSF fluxes are biased faint (dev7 T08676 calibrators: +0.056
mag at 0.2-0.5% lobes, +0.2..+0.7 mag deeper; the delivered 21-tile table: 0.7-2.5
mag) while the row is flagged reliable. The zone is now marked per band. R06.
``FLAGS_{band}_fit`` carries the band's reasons (no data, PSF lobe, saturation =
R11; ZP extrapolation is reserved for R08); ``n_bands_usable`` counts the clean
bands and a fitted source with none is not reliable.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tract7dt.fit.grid_gpu.band_flags import (
    FLAG_NO_DATA,
    FLAG_PSF_LOBE,
    PSF_LOBE_FRAC,
    compose_band_flags,
    psf_lobe_zone,
)
from tract7dt.fit.grid_gpu.psf_polynomial import PsfPolynomial
from tract7dt.merge import compute_reliability_flag

S = 31
W, H = 1000.0, 800.0


def _gauss(sig):
    yy, xx = np.mgrid[:S, :S] - S // 2
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sig ** 2)
    return g / g.sum()


def _psf(coefs):
    return PsfPolynomial(coefs=coefs.astype(np.float32), x_origin=W / 2,
                         x_scale=W / 2, y_origin=H / 2, y_scale=H / 2,
                         H_psf=S, W_psf=S)


def _psf_breaking_outside():
    # stamp(X) = g_narrow + X^2 (g_wide - g_narrow), X = (x - 500) / 500.
    # Inside the image X^2 <= 1: a positive blend. Far out (X^2 = 9):
    # 9 g_wide - 8 g_narrow, negative at the centre -> a lobe.
    gn, gw = _gauss(1.5), _gauss(4.0)
    coefs = np.zeros((S, S, 6))                # order 2: 1, X, Y, X^2, XY, Y^2
    coefs[..., 0] = gn
    coefs[..., 3] = gw - gn
    return _psf(coefs)


def test_zone_marks_the_broken_extrapolation_only():
    psf = _psf_breaking_outside()
    x = np.array([300.0, 700.0, 2000.0, -1000.0, np.nan, 2000.0])
    y = np.array([400.0, 400.0, 400.0, 400.0, 400.0, np.nan])
    zone = psf_lobe_zone(psf, x, y)
    # inside the image (positive blend); far out on both sides (lobe);
    # non-finite positions are never flagged
    assert zone.tolist() == [False, False, True, True, False, False]


def test_zone_threshold_is_two_per_mille_of_the_peak():
    assert PSF_LOBE_FRAC == 0.002
    gn = _gauss(1.5)
    x = np.array([500.0]); y = np.array([400.0])
    for dip, flagged in ((0.001, False), (0.003, True)):
        c = np.zeros((S, S, 1)); c[..., 0] = gn
        c[0, 0, 0] = -dip * gn.max()
        assert psf_lobe_zone(_psf(c), x, y).tolist() == [flagged]
    c = np.zeros((S, S, 1)); c[..., 0] = -gn          # no positive peak
    assert psf_lobe_zone(_psf(c), x, y).tolist() == [True]


def test_zone_chunking_is_invisible():
    psf = _psf_breaking_outside()
    rng = np.random.default_rng(3)
    x = rng.uniform(-2000, 3000, 500)
    y = rng.uniform(0, 800, 500)
    a = psf_lobe_zone(psf, x, y, chunk=7)
    b = psf_lobe_zone(psf, x, y)
    np.testing.assert_array_equal(a, b)
    assert a.any() and not a.all()


def test_compose_band_flags_bits_and_usable_count():
    nan = np.nan
    out = pd.DataFrame({
        "FLUX_g_fit": [1.0, nan, 3.0, 4.0],
        "FLUXERR_g_fit": [0.1, nan, 0.3, 0.4],
        "FLUX_r_fit": [1.0, 2.0, nan, 4.0],
        "FLUXERR_r_fit": [0.1, 0.2, nan, 0.4],
    })
    zone_r = np.array([False, True, True, False])
    counts = compose_band_flags(out, ["g", "r"], psf_zone={"r": zone_r})
    assert out["FLAGS_g_fit"].dtype == np.int16
    assert out["FLAGS_g_fit"].tolist() == [0, FLAG_NO_DATA, 0, 0]
    assert out["FLAGS_r_fit"].tolist() == [
        0, FLAG_PSF_LOBE, FLAG_NO_DATA | FLAG_PSF_LOBE, 0]
    assert out["n_bands_usable"].tolist() == [2, 0, 1, 2]
    # psf_lobe counted only on otherwise measured cells (row 1)
    assert counts == {"g": {"no_data": 1, "psf_lobe": 0, "saturated": 0,
                            "partial_coverage": 0},
                      "r": {"no_data": 1, "psf_lobe": 1, "saturated": 0,
                            "partial_coverage": 0}}


def test_no_usable_band_hard_fails_a_fitted_source_only():
    df = pd.DataFrame({
        "excluded_any": [False, False, True, False],
        "x_pix_white_fit": [1.0, 2.0, np.nan, np.nan],
        "n_bands_usable": [3, 0, 0, 0],
    })
    compute_reliability_flag(df)
    assert df["fit_is_reliable"].tolist() == [True, False, False, False]
    issues = df["fit_quality_issues"].tolist()
    assert issues[1] == "no_usable_band"
    # excluded / unfitted rows keep their own reason only
    assert "no_usable_band" not in issues[2] and "no_usable_band" not in issues[3]


def test_reliability_unchanged_without_the_band_count_column():
    df = pd.DataFrame({"excluded_any": [False], "x_pix_white_fit": [1.0]})
    compute_reliability_flag(df)
    assert df["fit_is_reliable"].tolist() == [True]


def test_psf_npz_keeps_training_stars_and_qc_is_returned(tmp_path):
    from tract7dt.fit.grid_gpu.psf_pipeline_stage import (
        _save_one_psf, load_psf_polynomial_for_band)
    psf = _psf_breaking_outside()
    rng = np.random.default_rng(1)
    sx, sy = rng.uniform(100, 900, 5), rng.uniform(100, 700, 5)
    result = SimpleNamespace(psf_polynomial=psf, polynomial_order=2,
                             star_x=sx, star_y=sy)
    qc = _save_one_psf(tmp_path, "g", result, save_diagnostics=False)
    assert set(qc) == {"integral_p10", "integral_p90"}
    assert np.isfinite(qc["integral_p10"]) and np.isfinite(qc["integral_p90"])
    with np.load(tmp_path / "g" / "psf_polynomial.npz") as d:
        np.testing.assert_array_equal(d["star_x"], sx)
        np.testing.assert_array_equal(d["star_y"], sy)
    np.testing.assert_array_equal(
        load_psf_polynomial_for_band(tmp_path, "g").coefs, psf.coefs)


def test_psf_cache_key_tracks_the_source_frame(tmp_path):
    from tract7dt.fit.grid_gpu.psf_pipeline_stage import _psf_cache_key
    f1, f2 = tmp_path / "a.fits", tmp_path / "b.fits"
    f1.write_bytes(b"x" * 10)
    f2.write_bytes(b"x" * 10)
    cfg = {"image_scaling": {"zp_ref": 25.0}, "crop": {"margin": 100},
           "source_saturation_cut": {"saturation_divisor": 1.3}}
    ps = {"polynomial_order": 4, "thresh_sigma": 5.0}
    bd = {"fp": str(f1), "img_scaled": np.zeros((4, 5), np.float32)}
    k1 = _psf_cache_key(cfg, ps, bd)
    # stable and JSON-round-trip equal (it is compared with a stored copy)
    assert k1 == _psf_cache_key(cfg, ps, bd)
    assert json.loads(json.dumps(k1)) == k1
    # another frame of the same band, or another PSF setting -> another key
    assert _psf_cache_key(cfg, ps, {**bd, "fp": str(f2)}) != k1
    assert _psf_cache_key(cfg, {**ps, "thresh_sigma": 4.0}, bd) != k1
    assert _psf_cache_key({**cfg, "crop": {"margin": 50}}, ps, bd) != k1
