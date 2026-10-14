"""Plan v3 stage 1, bundle 3 (R08, R14, R13).

R08. The spatial ZP (a 4th-order polynomial over the field) used to be evaluated
everywhere. Outside its calibrators' hull it extrapolated (2026-09 request
sample: -42 to +274 mag, flagged reliable); inside the hull it overshot in
sparsely calibrated patches (0.11-1.5 mag). Now a row outside the kept
calibrators' hull, or where the model differs from the median of its k nearest
calibrators by more than 0.1 mag, gets the scalar ZP, the scalar's ZP error and
``FLAGS_{band}_fit`` bit 4; the band then no longer counts as usable. A
calibrator whose band value carries a reason flag does not calibrate that band.
The surface is saved in ``ZP/zp_model.json``.

R14. ``MAG_UL5_MARG_{band}_fit`` is the 5-sigma limit from the recommended error
(FLUXERR_MARG_TOT); the all-NaN MAG(ERR)_MARG_FRAC_* columns are gone; the
source-Poisson gain uses the configured ``image_scaling.zp_ref``.

R13. ``run_provenance.json`` records the code, the effective configuration, the
environment variables that can change results, and the schema version.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from tract7dt.fit.grid_gpu.band_flags import FLAG_PSF_LOBE, FLAG_ZP_EXTRAP
from tract7dt.zp import K_MAG, ZP_SUPPORT_TOL_MAG, _zp_support, compute_zp

BAND = "m500"


def _grid(n=8, lo=100.0, hi=900.0):
    g = np.linspace(lo, hi, n)
    x, y = np.meshgrid(g, g)
    return x.ravel(), y.ravel()


def test_support_flags_outside_hull_and_overshoot_only():
    kx, ky = _grid()
    kd = np.zeros(kx.size)                       # calibrators: flat ZP
    qx = np.array([500.0, 500.0, 2000.0, np.nan, 150.0])
    qy = np.array([500.0, 300.0, 2000.0, 10.0, 150.0])
    # the model: flat, except an overshoot of 0.3 mag at the 2nd query and a
    # small (below tolerance) deviation at the 5th
    delta = np.array([0.0, 0.3, 5.0, 0.0, 0.05])
    s = _zp_support(delta, kx, ky, kd, qx, qy, k=10, tol=ZP_SUPPORT_TOL_MAG)
    assert s["outside_hull"].tolist() == [False, False, True, False, False]
    assert s["mismatch"].tolist() == [False, True, False, False, False]
    assert len(s["hull"]) == 4                   # the square's corners


def test_support_without_a_hull_marks_every_finite_query():
    s = _zp_support(np.zeros(2), [0.0, 1.0], [0.0, 1.0], [0.0, 0.0],
                    np.array([0.5, np.nan]), np.array([0.5, 0.0]), k=5, tol=0.1)
    assert s["outside_hull"].tolist() == [True, False]
    assert s["hull"] == []


def _catalog(rng):
    """64 Gaia calibrators on a grid with a ZP gradient of 0.1 mag across the
    field, one calibrator in a PSF-lobe cell reading 0.5 mag faint, and three
    science rows: inside the hull, far outside it, and a no-data cell."""
    kx, ky = _grid()
    n_cal = kx.size
    zp_true = 25.0 + 1.25e-4 * (kx - 500.0)
    gmag = rng.uniform(14.0, 17.0, n_cal)
    flux = 10 ** (-0.4 * (gmag - zp_true))
    flags = np.zeros(n_cal, dtype=np.int16)
    flags[5] = FLAG_PSF_LOBE
    flux[5] *= 10 ** (-0.4 * 0.5)
    ids = [f"gaia_{i}" for i in range(n_cal)]
    gsid = [str(1000 + i) for i in range(n_cal)]
    sci_x = np.array([500.0, 3000.0, 450.0])
    sci_y = np.array([500.0, 3000.0, 450.0])
    sci_flux = np.array([200.0, 300.0, np.nan])
    sci_ferr = np.array([2.0, 3.0, np.nan])
    x = np.concatenate([kx, sci_x])
    y = np.concatenate([ky, sci_y])
    f = np.concatenate([flux, sci_flux])
    fe = np.concatenate([0.005 * flux, sci_ferr])
    n = f.size
    cat = pd.DataFrame({
        "ID": ids + ["s_in", "s_out", "s_nodata"],
        "gaia_source_id": gsid + ["", "", ""],
        "x_pix_white_fit": x, "y_pix_white_fit": y,
        f"FLUX_{BAND}_fit": f, f"FLUXERR_{BAND}_fit": fe,
        f"FLUX_WIN_{BAND}_fit": f * 0.9,
        f"FLUXERR_TOT_{BAND}_fit": fe * 1.1,
        f"FLUXERR_MARG_{BAND}_fit": fe,
        f"FLUX_MARG_FRAC_{BAND}_fit": np.zeros(n),
        f"FLUXERR_MARG_TOT_{BAND}_fit": fe * 1.5,
        f"FLAGS_{BAND}_fit": np.concatenate([flags, [0, 0, 1]]).astype(np.int16),
        "n_bands_usable": np.concatenate([(flags == 0).astype(np.int16), [1, 1, 0]]),
        "fit_is_reliable": np.concatenate([np.ones(n_cal, bool), [True, True, False]]),
        "fit_quality_issues": [""] * n_cal + ["", "", "no_usable_band"],
    })
    gaia = pd.DataFrame({"source_id": gsid, f"mag_{BAND}": gmag})
    return cat, gaia


def _cfg(tmp_path, gaia_csv):
    return {
        "inputs": {"gaiaxp_synphot_csv": str(gaia_csv)},
        "outputs": {"work_dir": str(tmp_path)},
        "zp": {"clip_sigma": 3.0, "clip_max_iters": 10, "zp_err_method": "bright_std",
               "zp_err_snr_min": 100.0, "min_calibrator_snr": 20.0,
               "gaia_quality_filter": False, "spatial_zp_model": "poly",
               "spatial_zp_poly_order": 1, "spatial_zp_min_per_cell": 10,
               "catalog_write_workers": 1, "plot_workers": 1},
        "plotting": {"dpi": 30},
    }


@pytest.fixture
def zp_run(tmp_path):
    rng = np.random.default_rng(1)
    cat, gaia = _catalog(rng)
    gaia_csv = tmp_path / "gaia.csv"
    gaia.to_csv(gaia_csv, index=False)
    cfg = _cfg(tmp_path, gaia_csv)
    out = tmp_path / "output_catalog.csv"
    compute_zp(cfg=cfg, merged_catalog_path=out, catalog_df=cat.copy())
    return cfg, out, pd.read_csv(out).set_index("ID")


def test_zp_falls_back_and_flags_outside_the_calibrators(zp_run):
    cfg, out, c = zp_run
    zs = pd.read_csv(f"{cfg['outputs']['work_dir']}/ZP/zp_summary.csv").set_index("band")
    zp_med = zs.at[BAND, "zp_median"]
    f = c[f"FLAGS_{BAND}_fit"]
    assert f["s_out"] & FLAG_ZP_EXTRAP and not f["s_in"] & FLAG_ZP_EXTRAP
    # outside: the scalar ZP, not the gradient extrapolated 2500 px (+0.31 mag)
    assert c.at["s_out", f"MAG_{BAND}_fit"] == pytest.approx(zp_med - 2.5 * np.log10(300.0), abs=1e-9)
    # inside: the spatial model (x = 500 is the field centre of the gradient)
    assert c.at["s_in", f"MAG_{BAND}_fit"] == pytest.approx(25.0 - 2.5 * np.log10(200.0), abs=0.01)
    # the fallback row carries the scalar's ZP error, larger than the model's
    stat = K_MAG * 3.0 / 300.0
    zerr_out = np.sqrt(c.at["s_out", f"MAGERR_{BAND}_fit"] ** 2 - stat ** 2)
    assert zerr_out == pytest.approx(zs.at[BAND, "zp_err_fallback"], rel=1e-6)
    assert zs.at[BAND, "zp_err_fallback"] > zs.at[BAND, "zp_err_std"]
    assert zs.at[BAND, "n_zp_extrap_hull"] == 1
    # its only band is gone: no usable band -> not reliable
    assert c.at["s_out", "n_bands_usable"] == 0
    assert not c.at["s_out", "fit_is_reliable"]
    assert "no_usable_band" in c.at["s_out", "fit_quality_issues"]
    assert c.at["s_in", "fit_is_reliable"] and c.at["s_in", "n_bands_usable"] == 1
    # the no-data cell stays without a magnitude and is not flagged again
    assert np.isnan(c.at["s_nodata", f"MAG_{BAND}_fit"])
    assert c.at["s_nodata", f"FLAGS_{BAND}_fit"] == 1


def test_flagged_calibrator_does_not_calibrate(zp_run):
    cfg, _, _ = zp_run
    po = pd.read_csv(f"{cfg['outputs']['work_dir']}/ZP/zp_all_bands__per_object.csv", dtype={"gaia_source_id": str})
    row = po[po.gaia_source_id == "1005"].iloc[0]
    assert row.band_flags == FLAG_PSF_LOBE and not row.is_kept
    zs = pd.read_csv(f"{cfg['outputs']['work_dir']}/ZP/zp_summary.csv").set_index("band")
    assert zs.at[BAND, "n_calibrators_flag_excluded"] == 1


def test_ul5_marg_and_no_marg_frac_columns(zp_run):
    _, _, c = zp_run
    assert not [col for col in c.columns if "MARG_FRAC" in col and col.startswith("MAG")]
    assert not [col for col in c.columns if col.startswith("MAG_WIN")]
    zp_in = c.at["s_in", f"MAG_{BAND}_fit"] + 2.5 * np.log10(200.0)
    fe_mt = c.at["s_in", f"FLUXERR_MARG_TOT_{BAND}_fit"]
    fe_t = c.at["s_in", f"FLUXERR_TOT_{BAND}_fit"]
    assert c.at["s_in", f"MAG_UL5_MARG_{BAND}_fit"] == pytest.approx(zp_in - 2.5 * np.log10(5 * fe_mt), abs=1e-9)
    assert c.at["s_in", f"MAG_UL5_{BAND}_fit"] == pytest.approx(zp_in - 2.5 * np.log10(5 * fe_t), abs=1e-9)
    assert np.isnan(c.at["s_nodata", f"MAG_UL5_MARG_{BAND}_fit"])


def test_zp_model_file_reproduces_the_surface(zp_run):
    cfg, _, c = zp_run
    doc = json.load(open(f"{cfg['outputs']['work_dir']}/ZP/zp_model.json"))
    m = doc["bands"][BAND]
    assert m["kind"] == "poly" and m["order"] == 1 and len(m["hull"]) == 4
    assert doc["tolerance_mag"] == ZP_SUPPORT_TOL_MAG and doc["code"]["source_sha256"]
    (xc, xs), (yc, ys) = m["x_extent"], m["y_extent"]
    X, Y = (500.0 - xc) / xs, (500.0 - yc) / ys
    zp = m["zp_median"] + m["coef"][0] + m["coef"][1] * X + m["coef"][2] * Y
    assert c.at["s_in", f"MAG_{BAND}_fit"] == pytest.approx(zp - 2.5 * np.log10(200.0), abs=1e-9)


def test_rerun_on_its_own_output_is_identical(zp_run, tmp_path):
    cfg, out, c1 = zp_run
    first = pd.read_csv(out)
    # the CLI path (`tract7dt compute-zp`): reads the written catalog back
    compute_zp(cfg=cfg, merged_catalog_path=out)
    second = pd.read_csv(out)
    assert list(second.columns) == list(first.columns)
    pd.testing.assert_frame_equal(second, first)


def test_source_poisson_gain_uses_configured_zp_ref(tmp_path):
    from astropy.io import fits

    from tract7dt.fit.grid_gpu.error_columns import _source_poisson_gains
    fp = tmp_path / f"calib_7DT01_T00001_{BAND}_com.fits"
    fits.PrimaryHDU(np.zeros((2, 2), np.float32),
                    header=fits.Header({"ZP_AUTO": 27.0, "EGAIN": 10.0})).writeto(fp)
    lst = tmp_path / "image_list.txt"
    lst.write_text(f"{fp}\n")
    log = logging.getLogger("test")
    g = {}
    for zp_ref in (25.0, 23.0):
        cfg = {"inputs": {"image_list_file": str(lst)}, "image_scaling": {"zp_ref": zp_ref}}
        g[zp_ref] = _source_poisson_gains(cfg, [BAND], {}, log)[BAND]
    # internal counts = native ADU * 10**((zp_ref - ZP_AUTO)/2.5)
    assert g[25.0] == pytest.approx(10.0 / 10 ** ((25.0 - 27.0) / 2.5))
    assert g[23.0] / g[25.0] == pytest.approx(10 ** (2.0 / 2.5))


def test_run_provenance_records_code_config_and_environment(tmp_path, monkeypatch):
    from tract7dt.provenance import SCHEMA_VERSION, write_run_provenance
    monkeypatch.setenv("TRACT7DT_RENDER_STAMP_RENORM", "0")
    monkeypatch.setenv("SOME_SECRET_TOKEN", "x")
    cfg = {"outputs": {"work_dir": tmp_path}, "zp": {"clip_sigma": np.float64(3.0)},
           "compute": {"gpu": {"devices": (0, 1)}}}
    p = write_run_provenance(cfg, tmp_path / "w")
    doc = json.load(open(p))
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["config"]["zp"]["clip_sigma"] == 3.0
    assert doc["config"]["outputs"]["work_dir"] == str(tmp_path)
    assert doc["config"]["compute"]["gpu"]["devices"] == [0, 1]
    assert doc["environment"]["TRACT7DT_RENDER_STAMP_RENORM"] == "0"
    assert "SOME_SECRET_TOKEN" not in doc["environment"]
    code = doc["code"]
    assert len(code["source_sha256"]) == 64
    assert code["git_commit"] is None or len(code["git_commit"]) == 40
    assert "FLAGS_{band}_fit" in doc["missing_values"]
