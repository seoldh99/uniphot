"""Plan v3 stage 1, session 5 (R28, R29).

R28. The Gaia ZP calibrator candidates are every GaiaXP source on the (cropped)
image. They used to be the sources in a square box built around the input
catalog (at least ``zp.min_box_size_pix`` = 6000 px), so in a few-target run the
outermost targets always sat on the calibrators' hull edge -- R08 then put
bit 4 (ZP not supported) on almost every band and 20 of the 82 targets of the
2026-09 request sample had no usable band -- and the ZP of a star depended on
which other targets were run with it. The key is retired.

R29. With ``zp.gaia_quality_filter`` on, the Gaia quality flags (RUWE, BP/RP
excess) of every calibrator candidate are obtained before the fit, within one
time budget, or the run fails there. A failed fetch used to leave the ZP
calibrated on the unfiltered set while RUN_STATUS said "ok" (crowded T08285,
2026-09-27: 27 min hung, then calibrators +23-42%). ``zp.gaia_quality_autofetch``
(the switch into that path) is retired.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from tract7dt import gaia_quality as gq
from tract7dt import run_status, zp
from tract7dt.run_status import RunContractError

H_IMG, W_IMG = 1500, 2000


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [W_IMG / 2.0, H_IMG / 2.0]
    w.wcs.cdelt = [-0.505 / 3600.0, 0.505 / 3600.0]
    return w


def _gaia_grid(w, tmp_path):
    """Nine GaiaXP stars spread over the whole image, one just off it."""
    xs = np.array([200.0, 1000.0, 1800.0])
    ys = np.array([200.0, 750.0, 1300.0])
    gx, gy = np.meshgrid(xs, ys)
    gx = np.append(gx.ravel(), -50.0)
    gy = np.append(gy.ravel(), 700.0)
    ra, dec = w.all_pix2world(gx, gy, 0)
    gaia = pd.DataFrame({"source_id": [str(5000 + i) for i in range(gx.size)],
                         "ra": ra, "dec": dec,
                         "phot_g_mean_mag": 15.0, "bp_rp": 0.8, "mag_g": 15.0})
    p = tmp_path / "gaia.csv"
    gaia.to_csv(p, index=False)
    return p


def _augment(tmp_path, target_xy, extra_zp=None):
    w = _wcs()
    gaia_csv = _gaia_grid(w, tmp_path)
    tx = np.array([p[0] for p in target_xy], dtype=float)
    ty = np.array([p[1] for p in target_xy], dtype=float)
    ra, dec = w.all_pix2world(tx, ty, 0)
    cat = pd.DataFrame({"ID": [f"t{i}" for i in range(tx.size)], "RA": ra,
                        "DEC": dec, "TYPE": "STAR"})
    zp_cfg = {"gaia_mag_min": 4.0, "gaia_mag_max": 28.0,
              "match_radius_arcsec": 1.0, "inject_gaia_sources": True,
              "gaia_pos_err_mode": "fixed", "gaia_pos_err_pix": 0.1,
              "gaia_quality_filter": False, "save_gaia_overlay": False}
    zp_cfg.update(extra_zp or {})
    cfg = {"zp": zp_cfg,
           "inputs": {"gaiaxp_synphot_csv": str(gaia_csv),
                      "input_catalog": str(tmp_path / "input_catalog.csv")},
           "outputs": {"work_dir": str(tmp_path)},
           "image_scaling": {"zp_ref": 25.0},
           "compute": {"mode": "gpu"},
           "source_saturation_cut": {"enabled": False}}
    state = {"input_catalog": cat, "image_dict": {"g": {"wcs": w}},
             "white": np.zeros((H_IMG, W_IMG), dtype=np.float32)}
    aug = zp.augment_catalog_with_gaia(cfg=cfg, state=state)
    return set(aug.loc[aug["ID"].astype(str).str.startswith("gaia_"), "gaia_source_id"])


# ---------------------------------------------------------------- R28 ----

def test_gaia_in_image_is_the_whole_image():
    x = np.array([0.0, W_IMG - 0.01, -0.01, W_IMG, 10.0, np.nan])
    y = np.array([0.0, H_IMG - 0.01, 5.0, 5.0, H_IMG, 5.0])
    assert zp.gaia_in_image(x, y, img_h=H_IMG, img_w=W_IMG).tolist() == [
        True, True, False, False, False, False]


def test_calibrator_candidates_do_not_depend_on_the_input_catalog(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    corner = _augment(tmp_path / "a", [(100.0, 100.0), (130.0, 90.0)])
    centre = _augment(tmp_path / "b", [(1000.0, 700.0)])
    every_on_image = {str(5000 + i) for i in range(9)}
    assert corner == every_on_image          # not just the stars near the corner
    assert centre == every_on_image
    assert "5009" not in corner              # the one off the image stays out


def test_a_pinned_min_box_size_has_no_effect(tmp_path):
    # An old hand-built cfg may still carry the key; the selection ignores it
    # (load_config drops it with a note, test below).
    got = _augment(tmp_path, [(100.0, 100.0)], {"min_box_size_pix": 100})
    assert got == {str(5000 + i) for i in range(9)}


def test_retired_zp_keys_are_dropped_with_a_note(capsys):
    from tract7dt.config import DEFAULT_CONFIG, _drop_retired_keys
    for key in ("min_box_size_pix", "gaia_quality_autofetch"):
        assert key not in DEFAULT_CONFIG["zp"]
    user = {"zp": {"enabled": True, "min_box_size_pix": 6000,
                   "gaia_quality_autofetch": True}}
    _drop_retired_keys(user)
    assert user == {"zp": {"enabled": True}}
    err = capsys.readouterr().err
    # one summary note naming every dropped key (card 018)
    assert "retired" in err
    assert "zp.min_box_size_pix" in err and "zp.gaia_quality_autofetch" in err


# ---------------------------------------------------------------- R29 ----

IDS = ["100", "200", "300"]


def _flags(ids):
    return pd.DataFrame({"source_id": [str(i) for i in ids], "ruwe": 1.0,
                         "phot_bp_rp_excess_factor": 1.2})


def test_batch_fetch_stops_at_the_deadline(monkeypatch):
    seen_timeouts: list[float] = []

    def _down(url, data=None, timeout=None):
        seen_timeouts.append(timeout)
        raise OSError("archive down")

    monkeypatch.setattr(gq.urllib.request, "urlopen", _down)
    t0 = time.monotonic()
    with pytest.raises(gq.QualityFetchTimeout):
        gq.fetch_gaia_quality_flags(IDS, retries=50,
                                    deadline=time.monotonic() + 1.0)
    assert time.monotonic() - t0 < 5.0      # not 50 retries x backoff
    assert seen_timeouts and max(seen_timeouts) <= 1.0 + 1e-6


def _start_prefetch(tmp_path, *, budget=5.0):
    cache = tmp_path / "ZP" / "gaia_quality_cache.csv"
    deadline = time.monotonic() + budget
    t = gq.prefetch_quality_flags_async(IDS, cache_path=str(cache),
                                        deadline=deadline)
    zp._GAIA_QF_PREFETCH.clear()
    zp._GAIA_QF_PREFETCH.update(thread=t, ids=list(IDS),
                                cache_path=str(cache), deadline=deadline)
    return cache


CFG_ON = {"zp": {"enabled": True, "gaia_quality_filter": True}}


def test_failed_fetch_fails_the_run_before_the_fit(monkeypatch, tmp_path):
    def _boom(source_ids, **kwargs):
        raise OSError("Remote end closed connection without response")

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _boom)
    _start_prefetch(tmp_path)
    with pytest.raises(RunContractError) as ei:
        zp.require_gaia_quality_flags(CFG_ON, {})
    assert ei.value.kind == "gaia_quality_unavailable"
    assert "Remote end closed connection" in ei.value.message
    zp._GAIA_QF_PREFETCH.clear()


def test_flags_in_hand_pass_and_are_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(gq, "fetch_gaia_quality_flags",
                        lambda source_ids, **kw: _flags(source_ids))
    run_status.reset()
    cache = _start_prefetch(tmp_path)
    zp.require_gaia_quality_flags(CFG_ON, {})
    assert set(pd.read_csv(cache, dtype={"source_id": str})["source_id"]) == set(IDS)
    rec = run_status._REPORT["summary"]["gaia_quality_flags"]
    assert rec["n_candidates"] == 3 and rec["prefetch_error"] is None
    zp._GAIA_QF_PREFETCH.clear()


def test_nothing_to_obtain_when_filter_off_or_csv_has_flags(monkeypatch, tmp_path):
    def _never(*a, **k):
        raise AssertionError("must not fetch")

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _never)
    zp._GAIA_QF_PREFETCH.clear()                 # CSV carried the columns
    zp.require_gaia_quality_flags(CFG_ON, {})
    zp._GAIA_QF_PREFETCH.update(ids=IDS, cache_path=str(tmp_path / "c.csv"),
                                deadline=time.monotonic() + 5, thread=None)
    zp.require_gaia_quality_flags(
        {"zp": {"enabled": True, "gaia_quality_filter": False}}, {})
    zp._GAIA_QF_PREFETCH.clear()


def _zp_inputs(tmp_path):
    """36 calibrators on a grid + one science row; GaiaXP CSV without the
    quality columns, no cache in the work dir."""
    g = np.linspace(100.0, 900.0, 6)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    n = x.size
    rng = np.random.default_rng(3)
    gmag = rng.uniform(14.0, 17.0, n)
    flux = 10 ** (-0.4 * (gmag - 25.0))
    gsid = [str(9000 + i) for i in range(n)]
    cat = pd.DataFrame({
        "ID": [f"gaia_{s}" for s in gsid] + ["sci"],
        "gaia_source_id": gsid + [""],
        "x_pix_white_fit": np.append(x, 500.0), "y_pix_white_fit": np.append(y, 500.0),
        "FLUX_m500_fit": np.append(flux, 100.0),
        "FLUXERR_m500_fit": np.append(0.005 * flux, 1.0),
        "FLUXERR_TOT_m500_fit": np.append(0.006 * flux, 1.1),
        "FLUXERR_MARG_TOT_m500_fit": np.append(0.007 * flux, 1.2),
        "FLAGS_m500_fit": np.zeros(n + 1, dtype=np.int16),
        "n_bands_usable": np.ones(n + 1, dtype=np.int16),
        "fit_is_reliable": True, "fit_quality_issues": "",
    })
    gaia = pd.DataFrame({"source_id": gsid, "mag_m500": gmag,
                         "phot_g_mean_mag": gmag, "bp_rp": 0.8})
    gaia_csv = tmp_path / "gaia.csv"
    gaia.to_csv(gaia_csv, index=False)
    cfg = {"inputs": {"gaiaxp_synphot_csv": str(gaia_csv)},
           "outputs": {"work_dir": str(tmp_path)},
           "zp": {"clip_sigma": 3.0, "clip_max_iters": 10,
                  "zp_err_method": "bright_std", "zp_err_snr_min": 100.0,
                  "min_calibrator_snr": 20.0, "gaia_quality_filter": True,
                  "spatial_zp_model": "scalar", "catalog_write_workers": 1,
                  "plot_workers": 1},
           "plotting": {"dpi": 30}}
    return cfg, cat


def test_compute_zp_never_calibrates_on_the_unfiltered_set(monkeypatch, tmp_path):
    def _boom(source_ids, **kwargs):
        raise OSError("archive down")

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _boom)
    cfg, cat = _zp_inputs(tmp_path)
    out = tmp_path / "output_catalog.csv"
    with pytest.raises(RunContractError) as ei:
        zp.compute_zp(cfg=cfg, merged_catalog_path=out, catalog_df=cat.copy())
    assert ei.value.kind == "zp_not_computed"
    assert "archive down" in ei.value.message
    written = pd.read_csv(out)
    assert not any(c.startswith("MAG_") for c in written.columns)


def test_compute_zp_reads_the_cache_filled_before_the_fit(monkeypatch, tmp_path):
    def _never(*a, **k):
        raise AssertionError("must not fetch")

    monkeypatch.setattr(gq, "fetch_gaia_quality_flags", _never)
    cfg, cat = _zp_inputs(tmp_path)
    flags = _flags(cat["gaia_source_id"].iloc[:-1])
    flags.loc[0, "ruwe"] = 3.0                  # one calibrator to exclude
    (tmp_path / "ZP").mkdir()
    flags.to_csv(tmp_path / "ZP" / "gaia_quality_cache.csv", index=False)
    out = tmp_path / "output_catalog.csv"
    zp.compute_zp(cfg=cfg, merged_catalog_path=out, catalog_df=cat.copy())
    summ = pd.read_csv(tmp_path / "ZP" / "zp_summary.csv")
    assert bool(summ["gaia_quality_filter_active"].iloc[0])
    assert int(summ["gaia_n_calibrators_excluded"].iloc[0]) == 1
