"""Input-side band contract: one frame per band (REGISTER R10) and saturation
per band (R11).

R10. ``image_dict`` is keyed by band, so when a band was listed twice the LAST
entry silently won (83.9% of the campaign configs carry a duplicate band). The
deepest frame by header now wins, independent of list order, and the choice is
written to ``frames_used.csv``.

R11. A saturated pixel within ``radius_pix`` in ANY band used to drop the
source from the fit in ALL bands (dev7: 260 of 5,751 sources on T08676, 1,135
of 215,886 on T08285). Now the affected band is excluded (``_sat_{band}`` ->
``FLAGS`` bit 8, values blanked) and the source leaves the fit only when every
band is affected. The injected Gaia calibrators follow the same rule.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.wcs import WCS

from tract7dt.fit.grid_gpu.band_flags import (
    FLAG_NO_DATA,
    FLAG_SATURATED,
    compose_band_flags,
)
from tract7dt.load_inputs import (
    _filter_sources_near_saturation,
    _select_frames_per_band,
)


def _h(band, ul5=np.nan, zp=25.0, sky=1.0, tel="7DT01"):
    return {"band": band, "ul5_5": ul5, "zp_auto": zp, "skysig": sky,
            "telescop": tel}


def test_single_frames_are_all_used_in_list_order():
    frames = [(Path("/d/a_g.fits"), _h("g", 19.0)),
              (Path("/d/b_r.fits"), _h("r", 18.5))]
    chosen, table = _select_frames_per_band(frames)
    assert chosen == [Path("/d/a_g.fits"), Path("/d/b_r.fits")]
    assert table["used"].tolist() == [True, True]
    assert table["n_frames_band"].tolist() == [1, 1]


def test_deepest_duplicate_wins_whatever_the_list_order():
    shallow = (Path("/d/7DT12_m850.fits"), _h("m850", 16.842, tel="7DT12"))
    deep = (Path("/d/7DT11_m850.fits"), _h("m850", 16.928, tel="7DT11"))
    other = (Path("/d/7DT16_g.fits"), _h("g", 19.1))
    for frames in ([other, deep, shallow], [other, shallow, deep]):
        chosen, table = _select_frames_per_band(frames)
        # band order = first appearance; the m850 slot holds the deep frame
        assert chosen == [other[0], deep[0]]
        t = table.set_index("file")
        assert bool(t.loc[str(deep[0]), "used"])
        assert not bool(t.loc[str(shallow[0]), "used"])
        assert set(t["depth_key"]) == {"UL5_5"}


def test_exact_tie_goes_to_the_lexically_first_path():
    a = (Path("/d/a.fits"), _h("m800", 17.4))
    b = (Path("/d/b.fits"), _h("m800", 17.4))
    for frames in ([a, b], [b, a]):
        chosen, _ = _select_frames_per_band(frames)
        assert chosen == [a[0]]


def test_sky_depth_is_used_when_a_frame_lacks_ul5():
    # sky depth = ZP_AUTO - 2.5 log10(SKYSIG): 19.50 + 2.5 = 22.00 beats
    # 19.60 + 2.0 (SKYSIG 0.158) = 21.60, although UL5_5 would say otherwise
    a = (Path("/d/a.fits"), _h("m800", 17.0, zp=19.50, sky=0.1))
    b = (Path("/d/b.fits"), _h("m800", np.nan, zp=19.60, sky=0.158489))
    chosen, table = _select_frames_per_band([b, a])
    assert chosen == [a[0]]
    assert set(table["depth_key"]) == {"ZP_AUTO-2.5log10(SKYSIG)"}
    np.testing.assert_allclose(sorted(table["depth_mag"]), [21.6, 22.0],
                               atol=1e-4)


def _wcs(nx=200, ny=150):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-0.5 / 3600, 0.5 / 3600]
    return w


def _image_dict(sat_boxes, nx=200, ny=150):
    w = _wcs(nx, ny)
    out = {}
    for band, boxes in sat_boxes.items():
        m = np.zeros((ny, nx), bool)
        for (x0, x1, y0, y1) in boxes:
            m[y0:y1, x0:x1] = True
        out[band] = {"wcs": w, "satur_mask": m}
    return out


def _catalog_at(w, xy):
    ra, dec = w.all_pix2world(np.array([p[0] for p in xy], float),
                              np.array([p[1] for p in xy], float), 0)
    return pd.DataFrame({"ID": [f"s{i}" for i in range(len(xy))],
                         "RA": ra, "DEC": dec})


def test_saturation_is_per_band_and_excludes_only_when_every_band_is_hit():
    # source 0 near saturation only in g; source 1 in g and r; source 2 clean
    idict = _image_dict({"g": [(48, 53, 48, 53), (148, 153, 98, 103)],
                         "r": [(148, 153, 98, 103)]})
    w = idict["g"]["wcs"]
    cat = _catalog_at(w, [(50, 50), (150, 100), (100, 30)])
    out = _filter_sources_near_saturation(
        input_catalog=cat, image_dict=idict, ra_col="RA", dec_col="DEC",
        radius_pix=5.0)
    assert out["_sat_g"].tolist() == [True, True, False]
    assert out["_sat_r"].tolist() == [False, True, False]
    assert out["excluded_saturation"].tolist() == [False, True, False]


def test_saturation_respects_the_candidate_mask():
    idict = _image_dict({"g": [(48, 53, 48, 53)], "r": [(48, 53, 48, 53)]})
    cat = _catalog_at(idict["g"]["wcs"], [(50, 50), (50, 50)])
    out = _filter_sources_near_saturation(
        input_catalog=cat, image_dict=idict, ra_col="RA", dec_col="DEC",
        radius_pix=5.0, candidate_mask=np.array([True, False]))
    assert out["excluded_saturation"].tolist() == [True, False]
    assert out["_sat_g"].tolist() == [True, False]


def test_injected_gaia_follow_the_same_rule():
    from tract7dt.zp import _filter_gaia_saturation
    idict = _image_dict({"g": [(48, 53, 48, 53), (148, 153, 98, 103)],
                         "r": [(148, 153, 98, 103)]})
    w = idict["g"]["wcs"]
    c = _catalog_at(w, [(50, 50), (150, 100), (100, 30)])
    gaia = pd.DataFrame({"source_id": [1, 2, 3], "ra": c["RA"],
                         "dec": c["DEC"]})
    kept, sx, sy, hits = _filter_gaia_saturation(
        gaia, image_dict=idict, wcs=w, radius_pix=5.0)
    assert kept["source_id"].tolist() == [1, 3]          # 2: every band hit
    assert len(sx) == len(sy) == 1
    assert hits["g"].tolist() == [True, False]           # aligned with kept
    assert hits["r"].tolist() == [False, False]


def test_saturated_cells_are_blanked_and_flagged():
    out = pd.DataFrame({
        "FLUX_g_fit": [10.0, 20.0, np.nan],
        "FLUXERR_g_fit": [1.0, 2.0, np.nan],
        "FLUX_WIN_g_fit": [9.0, 18.0, np.nan],
        "FLUX_r_fit": [5.0, 6.0, 7.0],
        "FLUXERR_r_fit": [0.5, 0.6, 0.7],
    })
    sat_g = np.array([True, False, True])
    counts = compose_band_flags(out, ["g", "r"], saturated={"g": sat_g})
    assert out["FLAGS_g_fit"].tolist() == [
        FLAG_SATURATED, 0, FLAG_NO_DATA | FLAG_SATURATED]
    assert np.isnan(out.loc[0, "FLUX_g_fit"])
    assert np.isnan(out.loc[0, "FLUXERR_g_fit"])
    assert np.isnan(out.loc[0, "FLUX_WIN_g_fit"])
    assert out.loc[1, "FLUX_g_fit"] == 20.0
    assert out["FLAGS_r_fit"].tolist() == [0, 0, 0]
    assert out["n_bands_usable"].tolist() == [1, 2, 1]
    assert counts["g"]["saturated"] == 1                 # measured cells only


def test_retired_saturation_switch_is_dropped_with_a_note(tmp_path, capsys):
    from tract7dt.config import DEFAULT_CONFIG, _drop_retired_keys
    assert "require_all_bands" not in DEFAULT_CONFIG["source_saturation_cut"]
    user = {"source_saturation_cut": {"enabled": True,
                                      "require_all_bands": False}}
    _drop_retired_keys(user)
    assert user == {"source_saturation_cut": {"enabled": True}}
    err = capsys.readouterr().err  # one summary note naming the key (card 018)
    assert "retired" in err and "source_saturation_cut.require_all_bands" in err


def test_calibrators_must_be_clean_in_every_band():
    # R11 side effect guard: a source with FLAGS bit 8 in ANY band must not
    # calibrate any band (T08676: such bright stars took 84-99% of the ZP
    # weight and moved ZPs by up to 26 mmag).
    from tract7dt.zp import saturated_in_any_band
    cat = pd.DataFrame({"FLAGS_g_fit": np.array([0, 8, 1, 0], np.int16),
                        "FLAGS_r_fit": np.array([2, 0, 0, 9], np.int16),
                        "FLUX_g_fit": [1.0, 2.0, 3.0, 4.0]})
    assert saturated_in_any_band(cat).tolist() == [False, True, False, True]
    assert saturated_in_any_band(cat[["FLUX_g_fit"]]).tolist() == [False] * 4
