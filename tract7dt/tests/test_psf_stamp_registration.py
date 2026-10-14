"""Registration of PSF training stamps (M2 unit a.3; design v2.5 3rd ed. 5.5, 6.4, 6.5).

The training stamp is moved to its own centre by a Fourier sub-pixel shift of the 61 px cutout (the
pixels the stamp is read from), zero-padded by 16 px on every side (3rd ed. 6.5), then cut to the
central 55 px -- not by Lanczos-3, whose response above 1 at middle frequencies sharpens the mean
registered stamp (+0.23 % at the peak for FWHM 3.7-4.5 px).

Expected values (designer, ``v25/reg_edge_check.py``, "61 px cutout shifted, then the central 55 px",
pad 64, true centres, 150 random phases): mean registered peak - 1 = -0.031 %, -0.004 %, -0.000 % and
PSF-fit flux bias of the phase-0 truth -0.0013, -0.0002, -0.0003 mmag at FWHM 3.7, 4.5, 6.0.  Here the
engine path also estimates the centre (gw centroid, noiseless: rms 0.003 px) -- allowed: +-0.01 % on
the peak, |flux bias| <= 0.01 mmag (the allowances of round 2, unchanged).

History (impl_a.3.md): in round 2 the test star was rendered only inside its 55 px box (the synth box
convention), so the 61 px cutout held a light edge at 27.5 px; the Fourier shift rang there and the
FWHM 6.0 flux bias was +0.0115 mmag against the 0.01 allowance -- FAILED (commit 2d4a48c, seed 7).
This version renders the star on 75 px, so the cutout is filled to its edge as on a real image; it
was pre-registered with a new seed (41) in a test-only commit before its first run.
"""
from __future__ import annotations

import numpy as np
import pytest

import synth
from tract7dt.fit.grid_gpu import psf_from_stars as pfs


def _reference_fourier_shift(img, sx, sy, pad=16):
    """The definition, written independently: content moved by (sx, sy), zero padding ``pad``."""
    n0, n1 = img.shape
    p = np.zeros((n0 + 2 * pad, n1 + 2 * pad))
    p[pad:pad + n0, pad:pad + n1] = img
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.fftfreq(p.shape[1])[None, :]
    out = np.fft.ifft2(np.fft.fft2(p) * np.exp(-2j * np.pi * (fx * sx + fy * sy))).real
    return out[pad:pad + n0, pad:pad + n1]


def test_fourier_shift_follows_its_definition():
    rng = np.random.default_rng(3)
    img = rng.normal(size=(61, 61))
    for sx, sy in [(0.0, 0.0), (0.37, -0.21), (-0.5, 0.5), (1.0, 0.0)]:
        got = pfs.fourier_shift(img, sx, sy)
        np.testing.assert_allclose(got, _reference_fourier_shift(img, sx, sy), rtol=0, atol=1e-12)
    np.testing.assert_allclose(pfs.fourier_shift(img, 0.0, 0.0), img, rtol=0, atol=1e-13)


#                    FWHM, designer's 61-px-crop peak bias (%)
@pytest.mark.parametrize("fwhm, want_peak_pct", [(3.7, -0.031), (4.5, -0.004), (6.0, -0.000)])
def test_registered_stamp_is_not_sharpened(fwhm, want_peak_pct):
    psf = synth.PSFField((600, 600), fwhm=fwhm)
    t0 = synth.pixel_integrate(lambda u, v: psf.continuous(300.0, 300.0, u, v), 0.0, 0.0, size=75)
    truth = t0[10:65, 10:65] / t0[10:65, 10:65].sum()
    rng = np.random.default_rng(41)
    acc = np.zeros((55, 55))
    for _ in range(150):
        px, py = 300.0 + rng.uniform(-0.5, 0.5), 300.0 + rng.uniform(-0.5, 0.5)
        ix, iy = int(np.rint(px)), int(np.rint(py))
        st = synth.pixel_integrate(lambda u, v: psf.continuous(px, py, u, v), px - ix, py - iy, size=75)
        img = np.zeros((600, 600))
        img[iy - 37:iy + 38, ix - 37:ix + 38] = 1e5 * st
        s1, fx, fy, keep = pfs.extract_centered_stamps(
            img, np.ones_like(img), np.zeros(img.shape, bool), np.array([px]), np.array([py]),
            cutout_size=61, psf_size=55, star_local_bkg_subtract=False,
            centering_method="gw_centroid", fwhm_pix=fwhm)
        assert len(keep) == 1
        acc += s1[0].astype(np.float64)
    mean = acc / 150.0
    peak_pct = 100.0 * (mean[27, 27] / truth[27, 27] - 1.0)
    flux_mmag = -2.5e3 * np.log10((mean * truth).sum() / (mean * mean).sum())
    assert abs(peak_pct - want_peak_pct) <= 0.01, (fwhm, peak_pct)
    assert abs(flux_mmag) <= 0.01, (fwhm, flux_mmag)
