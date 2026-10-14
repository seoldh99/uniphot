"""Gate for the band-parallel raw-sky F_corr pass (efficiency arc, 2026-08-02).

The pass was serial and cost ~4.7 s/band (92-114 s per pipeline pass on a
20-22-band tile). It is now threaded. These tests exercise the ENGINE function
``_fcorr_raw_sky_pass`` itself -- not a re-implementation of it in the test
file, which is the failure mode that let the 2026-08-01b coverage-guard defect
through the earlier guard tests.

The contract being gated:
  1. every band's F_corr is BIT-IDENTICAL to calling the estimator serially,
  2. every diag field is bit-identical (NaN aligned with NaN),
  3. the results are ordered by BAND, never by completion,
  4. one band raising does not poison its neighbours,
  5. the worker count is DERIVED (work / cores / memory), never a constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from tract7dt.fit.grid_gpu.error_model import (
    _FCORR_MAX_CROP, measure_correlated_noise_factor)
from tract7dt.fit.grid_gpu.pipeline import (
    _fcorr_band_workers, _fcorr_raw_sky_pass)
from tract7dt.fit.grid_gpu.psf_polynomial import PsfPolynomial

SIDE = 384          # keeps the suite fast; the estimator's math is size-agnostic


def _psf(sigma=1.6, half=7):
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    stamp = np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    stamp /= stamp.sum()
    return PsfPolynomial(
        coefs=stamp[:, :, None].astype(np.float32),   # n_terms = 1 -> order 0
        x_origin=0.0, x_scale=1.0, y_origin=0.0, y_scale=1.0,
        H_psf=stamp.shape[0], W_psf=stamp.shape[1],
    )


def _band_image(seed, side=SIDE):
    """Correlated sky (the resampling signature the estimator accepts) + sources."""
    from scipy.signal import fftconvolve
    rng = np.random.default_rng(seed)
    k = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]], np.float32)
    k /= np.sqrt((k ** 2).sum())
    img = fftconvolve(rng.normal(0, 1, (side + 8, side + 8)).astype(np.float32),
                      k, mode="same")[:side, :side].astype(np.float32)
    ys = rng.integers(12, side - 12, 60)
    xs = rng.integers(12, side - 12, 60)
    img[ys, xs] += rng.uniform(20, 300, 60).astype(np.float32)
    bad = np.zeros((side, side), bool)
    bad[:3, :] = True
    return img, bad


def _scene(n_bands=6):
    bands = [f"m{400 + 25 * i}" for i in range(n_bands)]
    image_dict, psfs = {}, {}
    for i, bn in enumerate(bands):
        img, bad = _band_image(1000 + i)
        image_dict[bn] = {"img_scaled": img, "bad": bad}
        psfs[bn] = _psf()
    return bands, image_dict, psfs


def _serial_reference(bands, image_dict, psfs):
    from tract7dt.fit.grid_gpu.psf_polynomial import evaluate_psf_at_np
    out = {}
    diags = []
    for bn in bands:
        img = image_dict[bn]["img_scaled"]
        h, w = np.asarray(img).shape
        phi = evaluate_psf_at_np(psfs[bn], w * 0.5, h * 0.5)
        F, dg = measure_correlated_noise_factor(
            img, image_dict[bn]["bad"], phi, band=bn, log_fn=None)
        out[bn] = float(F)
        diags.append(dg)
    return out, diags


def _same_value(x, y):
    if isinstance(x, float) and isinstance(y, float):
        return x == y or (np.isnan(x) and np.isnan(y))
    return x == y


def test_parallel_matches_serial_bit_for_bit():
    bands, image_dict, psfs = _scene()
    ref_f, ref_d = _serial_reference(bands, image_dict, psfs)
    got_f, got_d = _fcorr_raw_sky_pass(bands, image_dict, psfs)

    assert list(got_f.keys()) == bands, "results must be in BAND order"
    for bn in bands:
        assert got_f[bn].hex() == ref_f[bn].hex(), f"{bn}: F_corr differs"
    assert len(got_d) == len(ref_d)
    for bn, dg, rg in zip(bands, got_d, ref_d):
        assert set(dg) == set(rg), f"{bn}: diag keys differ"
        for k in rg:
            assert _same_value(dg[k], rg[k]), f"{bn}.{k}: {dg[k]!r} != {rg[k]!r}"


def test_repeated_calls_are_bit_identical():
    """Threading must not make the pass depend on scheduling."""
    bands, image_dict, psfs = _scene()
    a, _ = _fcorr_raw_sky_pass(bands, image_dict, psfs)
    b, _ = _fcorr_raw_sky_pass(bands, image_dict, psfs)
    assert list(a) == list(b)
    for bn in bands:
        assert a[bn].hex() == b[bn].hex()


def test_one_bad_band_is_isolated():
    """A band that raises falls back to 1.0 without disturbing the others."""
    bands, image_dict, psfs = _scene(4)
    ref_f, _ = _serial_reference(bands, image_dict, psfs)
    psfs[bands[1]] = None                     # evaluate_psf_at_np will raise
    got_f, got_d = _fcorr_raw_sky_pass(bands, image_dict, psfs)

    assert got_f[bands[1]] == 1.0
    assert len(got_d) == len(bands) - 1       # the failed band contributes none
    for bn in (bands[0], bands[2], bands[3]):
        assert got_f[bn].hex() == ref_f[bn].hex()


def test_worker_count_is_derived_not_constant():
    bands, image_dict, psfs = _scene(6)
    n, why = _fcorr_band_workers(bands, image_dict)
    assert 1 <= n <= len(bands)
    assert "band(s)" in why and "core(s)" in why and "per band" in why

    # The memory term must be bounded by the CROP, not the raw image: a band
    # twice the crop in both axes must cost the same as one exactly at the
    # crop, because the estimator never looks outside its central crop.
    def _fake(shape):
        return {bn: {"img_scaled": np.broadcast_to(
            np.zeros(1, np.float32), shape)} for bn in bands}

    at_crop, why_at = _fcorr_band_workers(
        bands, _fake((_FCORR_MAX_CROP, _FCORR_MAX_CROP)))
    over_crop, why_over = _fcorr_band_workers(
        bands, _fake((_FCORR_MAX_CROP * 2, _FCORR_MAX_CROP * 2)))
    assert (at_crop, why_at) == (over_crop, why_over)

    # ... and a band SMALLER than the crop must cost strictly less, so the
    # bound really is min(shape, crop) and not the constant crop size.
    small, why_small = _fcorr_band_workers(
        bands, _fake((_FCORR_MAX_CROP // 8, _FCORR_MAX_CROP // 8)))
    assert why_small != why_at
    assert small >= at_crop

    assert _fcorr_band_workers([], {}) == (1, "no bands")
