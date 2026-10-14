"""Partial-coverage band flag (plan v3 stage 1, session 5: REGISTER R30).

A source on the edge of a band's footprint has part of its PSF on pixels
without data; its flux then leans on the PSF model over pixels nobody observed.
On Gaia-XP calibrators (S/N >= 20; dev8 session-4 runs) the error stops covering
the scatter once more than 10% of the PSF is unobserved (median |pull| x1.2-1.3
over fully covered cells, x2.4-2.7 above 45%). Such cells now carry
``FLAGS_{band}_fit`` bit 16 (value kept) and no longer count as usable; the
no-data mask used is the one before saturation and the bright mask are folded
in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tract7dt.fit.grid_gpu.band_flags import (
    FLAG_NO_DATA,
    FLAG_PARTIAL_COVERAGE,
    PARTIAL_COVERAGE_MAX,
    compose_band_flags,
    psf_uncovered_fraction,
)
from tract7dt.fit.grid_gpu.psf_polynomial import PsfPolynomial

S = 21
H, W = 200, 300


def _psf(sig=2.0):
    yy, xx = np.mgrid[:S, :S] - S // 2
    g = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sig ** 2)
    coefs = (g / g.sum())[..., None]           # order 0: the same stamp everywhere
    return PsfPolynomial(coefs=coefs.astype(np.float32), x_origin=W / 2,
                         x_scale=W / 2, y_origin=H / 2, y_scale=H / 2,
                         H_psf=S, W_psf=S)


def _brute(psf_stamp, nodata, x, y):
    h = S // 2
    pad = np.pad(nodata, h, constant_values=True)
    ix, iy = int(np.rint(x)), int(np.rint(y))
    win = pad[iy:iy + S, ix:ix + S]
    p = np.clip(psf_stamp, 0, None)
    return float((p * win).sum() / p.sum())


def test_fully_covered_interior_is_zero():
    psf = _psf()
    nodata = np.zeros((H, W), bool)
    u = psf_uncovered_fraction(psf, nodata, np.array([150.0, 40.0]),
                               np.array([100.0, 60.0]))
    assert u.tolist() == [0.0, 0.0]


def test_matches_brute_force_on_a_footprint_edge():
    psf = _psf()
    stamp = np.asarray(psf.coefs)[..., 0]
    nodata = np.zeros((H, W), bool)
    nodata[:, 200:] = True                      # the band's footprint ends at x = 200
    nodata[150:, :] = True                      # and at y = 150
    rng = np.random.default_rng(0)
    x = rng.uniform(0, W - 1, 300)
    y = rng.uniform(0, H - 1, 300)
    u = psf_uncovered_fraction(psf, nodata, x, y)
    ref = np.array([_brute(stamp, nodata, a, b) for a, b in zip(x, y)])
    np.testing.assert_allclose(u, ref, atol=1e-6)
    # a source centred on the edge has about half its PSF off the data
    half = psf_uncovered_fraction(psf, nodata, np.array([199.5 + 0.4]),
                                  np.array([60.0]))[0]
    assert 0.35 < half < 0.65


def test_image_border_counts_as_no_data_and_bad_positions():
    psf = _psf()
    nodata = np.zeros((H, W), bool)
    u = psf_uncovered_fraction(
        psf, nodata, np.array([0.0, 150.0, np.nan, -40.0]),
        np.array([100.0, 100.0, 5.0, 100.0]))
    assert 0.4 < u[0] < 0.6                     # half the stamp beyond the left border
    assert u[1] == 0.0
    assert u[2] == 0.0                          # non-finite position: never flagged
    assert u[3] == 1.0                          # off the image


def test_flag_keeps_the_value_and_drops_the_band_from_usable():
    out = pd.DataFrame({
        "FLUX_g_fit": [10.0, 20.0, np.nan], "FLUXERR_g_fit": [1.0, 2.0, np.nan],
        "FLUX_r_fit": [5.0, 6.0, 7.0], "FLUXERR_r_fit": [1.0, 1.0, 1.0],
    })
    part = {"g": np.array([False, True, True])}
    counts = compose_band_flags(out, ["g", "r"], partial=part)
    assert out["FLAGS_g_fit"].tolist() == [0, FLAG_PARTIAL_COVERAGE,
                                           FLAG_NO_DATA | FLAG_PARTIAL_COVERAGE]
    assert out.loc[1, "FLUX_g_fit"] == 20.0     # value kept
    assert out["n_bands_usable"].tolist() == [2, 1, 1]
    assert counts["g"]["partial_coverage"] == 1  # measured cells only
    assert PARTIAL_COVERAGE_MAX == 0.1


def test_compose_final_masks_keeps_the_no_data_part():
    from tract7dt.pipeline import _compose_final_masks
    bad = np.zeros((4, 12), bool)
    bad[:, 10:] = True                         # outside the footprint
    sat = np.zeros((4, 12), bool)
    sat[1, 3] = True
    bright = np.zeros((4, 12), bool)
    bright[2, 5] = True
    state = {"image_dict": {"g": {"bad": bad.copy(), "satur_mask": sat}},
             "bright_mask": bright}
    _compose_final_masks(state)
    d = state["image_dict"]["g"]
    kept = np.unpackbits(d["nodata_bits"], axis=1, count=12).astype(bool)
    np.testing.assert_array_equal(kept, bad)
    assert d["bad"][1, 3] and d["bad"][2, 5] and d["bad"][0, 11]
