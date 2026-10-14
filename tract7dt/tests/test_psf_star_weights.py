"""PSF training-star noise and weight, iteration 0 (M2 unit a.3).

Design: development design note design_v2.md (not distributed) 1.5 (v2.4, designer answer D6) and the
v2.5 weight definition (design_v2_5_answers.md: omega_s = u_s / (a_s + tau^2),
a_s = n_pix (sigma2_own,s + sigma2_nb,s) / F_s^2; the caller gives a_s and u_s); plan
impl_plan.md 1.3 a.3.  The expected values are computed here, pixel by pixel, from the design
formula -- not from the module under test:

  sigma2_own,s    = mean over the 55 x 55 box centred on (rint x_s, rint y_s) of sigma_pix^2
  sigma2_nb,s^(0) = mean over the same box of (sum_{j != s} F_j^SEP G(p - r_j))^2
                    (iteration 0 subtracts no neighbour, so its contamination is the
                    neighbours' whole light, not a subtraction uncertainty: D6)
  G               = unit-integral circular Gaussian with FWHM = the FWHM mode, at pixel centres
  a_s             = n_pix (sigma2_own,s + sigma2_nb,s) / F_s^2,  n_pix = 55^2
and for t >= 1 the neighbour amplitude is sigma_Fj (+) s_b F_j (quadrature), s_b^(-1) = 0.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tract7dt.fit.grid_gpu import psf_from_stars as pfs

HALF = 27
N_PIX = (2 * HALF + 1) ** 2
FWHM = 4.3


def _direct(star_x, star_y, star_flux, star_det, det_x, det_y, det_flux, sigma, fwhm):
    """The design formula, one pixel and one neighbour at a time (every neighbour, no reach)."""
    s2 = (fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))) ** 2
    a, own, nb = [], [], []
    for s in range(len(star_x)):
        ix, iy = int(np.rint(star_x[s])), int(np.rint(star_y[s]))
        own_sum = nb_sum = 0.0
        for py in range(iy - HALF, iy + HALF + 1):
            for px in range(ix - HALF, ix + HALF + 1):
                own_sum += float(sigma[py, px]) ** 2
                m = 0.0
                for j in range(len(det_x)):
                    if j == star_det[s]:
                        continue
                    r2 = (px - det_x[j]) ** 2 + (py - det_y[j]) ** 2
                    m += det_flux[j] * math.exp(-0.5 * r2 / s2) / (2.0 * math.pi * s2)
                nb_sum += m * m
        own.append(own_sum / N_PIX)
        nb.append(nb_sum / N_PIX)
        a.append(N_PIX * (own[-1] + nb[-1]) / star_flux[s] ** 2)
    return np.array(a), np.array(own), np.array(nb)


def _scene():
    """Six training stars among 30 detections: neighbours inside the box, at its edge,
    just outside it and far away; a sigma image that varies across every box."""
    rng = np.random.default_rng(31)
    h, w = 260, 400
    yy, xx = np.mgrid[0:h, 0:w]
    sigma = np.sqrt(1.0 + 0.4 * np.sin(xx / 13.0) ** 2 + 0.002 * yy)
    det_x = rng.uniform(5.0, 250.0, 30)        # no detection near star 5 (x = 360)
    det_y = rng.uniform(5.0, h - 5.0, 30)
    det_flux = 10.0 ** rng.uniform(2.0, 5.0, 30)
    star_det = np.array([3, 7, 11, 18, 22, 29])
    # training stars sit where their box fits in the image; neighbours on purpose:
    det_x[star_det] = [60.2, 150.7, 240.4, 70.6, 160.1, 360.9]
    det_y[star_det] = [60.8, 70.3, 65.5, 190.2, 180.6, 195.4]
    det_x[0], det_y[0], det_flux[0] = 60.2 + 9.0, 60.8 - 4.0, 3.0e4   # bright, inside star 0's box
    det_x[1], det_y[1] = 150.7 + 27.4, 70.3 + 3.0        # at the edge of star 1's box
    det_x[2], det_y[2] = 240.4 - 33.0, 65.5               # just outside star 2's box
    det_x[4], det_y[4] = 70.6 + 2.0, 190.2 + 2.5          # close to star 3
    star_flux = det_flux[star_det] * 1.07                  # a box sum, not the SEP flux
    return dict(star_x=det_x[star_det] + 0.05, star_y=det_y[star_det] - 0.04,
                star_flux=star_flux, star_det=star_det, det_x=det_x, det_y=det_y,
                det_flux=det_flux, sigma=sigma)


def test_iteration0_noise_matches_the_design_formula():
    sc = _scene()
    want_a, want_own, want_nb = _direct(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"],
                                        sc["det_x"], sc["det_y"], sc["det_flux"], sc["sigma"], FWHM)
    a, own, nb = pfs.psf0_star_noise(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"],
                                     sc["det_x"], sc["det_y"], sc["det_flux"], sc["sigma"], FWHM,
                                     half=HALF)
    np.testing.assert_allclose(own, want_own, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(nb, want_nb, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(a, want_a, rtol=1e-12, atol=0.0)
    # the scene has power: neighbour light is a large part of the noise of some stars and
    # negligible for others
    frac = nb / (own + nb)
    assert frac.max() > 0.5 and frac.min() < 1e-6


def test_isolated_star_has_its_own_noise_only():
    sigma = np.full((120, 120), 1.7)
    a, own, nb = pfs.psf0_star_noise(np.array([60.3]), np.array([59.6]), np.array([5000.0]),
                                     np.array([0]), np.array([60.3]), np.array([59.6]),
                                     np.array([4800.0]), sigma, FWHM, half=HALF)
    assert nb[0] == 0.0
    assert own[0] == pytest.approx(1.7 ** 2, rel=1e-14)
    assert a[0] == pytest.approx(N_PIX * 1.7 ** 2 / 5000.0 ** 2, rel=1e-14)


def _weight_with_neighbour(nb_flux, dx, dy, sigma=None):
    sigma = np.ones((200, 200)) if sigma is None else sigma
    a, _, _ = pfs.psf0_star_noise(np.array([100.2]), np.array([99.7]), np.array([2.0e4]),
                                  np.array([0]), np.array([100.2, 100.2 + dx]),
                                  np.array([99.7, 99.7 + dy]), np.array([2.1e4, nb_flux]),
                                  sigma, FWHM, half=HALF)
    return 1.0 / a[0]


def test_weight_falls_monotonically_and_without_jumps_as_neighbour_light_grows():
    fluxes = np.concatenate([[0.0], np.geomspace(1.0, 1e7, 57)])
    w = np.array([_weight_with_neighbour(f, 11.0, -6.0) for f in fluxes])
    assert np.all(np.diff(w) < 0.0), "the weight must fall strictly as the neighbour brightens"
    assert w[-1] < 1e-3 * w[0]
    # G7(b)-style perturbation: the response to a small change of the neighbour flux is linear
    # (no threshold, no jump) at every flux of the scan
    for f in (10.0, 1e3, 1e5):
        d1 = _weight_with_neighbour(f * (1 + 1e-6), 11.0, -6.0) - _weight_with_neighbour(f, 11.0, -6.0)
        d2 = _weight_with_neighbour(f * (1 + 2e-6), 11.0, -6.0) - _weight_with_neighbour(f, 11.0, -6.0)
        assert d1 < 0.0
        assert abs(d2 - 2.0 * d1) <= 1e-2 * abs(d1) + 1e-15 * _weight_with_neighbour(f, 11.0, -6.0)


def test_weight_is_continuous_as_a_neighbour_crosses_the_box_and_the_reach():
    """A neighbour moved from inside the box to far outside it: the weight follows the design
    formula (which has no cut) at every step, so no truncation shows as a jump."""
    sigma = np.ones((200, 200))
    for dx in np.arange(20.0, 121.0, 0.5):         # box edge 27.5, reach 27 + 39 sigma = 98
        got = 1.0 / pfs.psf0_star_noise(np.array([100.2]), np.array([99.7]), np.array([2.0e4]),
                                        np.array([0]), np.array([100.2, 100.2 + dx]),
                                        np.array([99.7, 99.7]), np.array([2.1e4, 1e8]),
                                        sigma, FWHM, half=HALF)[0][0]
        want = 1.0 / _direct(np.array([100.2]), np.array([99.7]), np.array([2.0e4]), np.array([0]),
                             np.array([100.2, 100.2 + dx]), np.array([99.7, 99.7]),
                             np.array([2.1e4, 1e8]), sigma, FWHM)[0][0]
        assert got == pytest.approx(want, rel=1e-12), dx


def test_neighbour_amplitude_of_later_iterations():
    sf = np.array([3.0, 0.0, 10.0])
    f = np.array([100.0, 50.0, -20.0])
    np.testing.assert_array_equal(pfs.neighbour_amplitude(sf, f), sf)           # s_b^(-1) = 0
    np.testing.assert_allclose(pfs.neighbour_amplitude(sf, f, 0.02),
                               np.sqrt(sf ** 2 + (0.02 * f) ** 2), rtol=1e-15)


# ---------------------------------------------------------------------------
# Round 2 (design v2.5 3rd ed. 5.2, 5.3): iteration 0 with the image's empirical stamp and the
# SEP AUTO flux; iterations t >= 1 with the independent sum of squares.
# ---------------------------------------------------------------------------

def _ref_shift(img, sx, sy, pad=16):
    """Fourier sub-pixel shift, written here from its definition (content moved by (sx, sy))."""
    n0, n1 = img.shape
    p = np.zeros((n0 + 2 * pad, n1 + 2 * pad))
    p[pad:pad + n0, pad:pad + n1] = img
    fy = np.fft.fftfreq(p.shape[0])[:, None]
    fx = np.fft.fftfreq(p.shape[1])[None, :]
    return np.fft.ifft2(np.fft.fft2(p) * np.exp(-2j * np.pi * (fx * sx + fy * sy))).real[pad:pad + n0, pad:pad + n1]


def _placed(stamp, x, y, shape):
    """A centred stamp placed with its centre at (x, y) (Fourier shift by the sub-pixel part),
    clipped to an image of ``shape``."""
    ix, iy = int(np.rint(x)), int(np.rint(y))
    sh = _ref_shift(stamp, x - ix, y - iy)
    h = stamp.shape[0] // 2
    pad = 2 * h + 2
    canvas = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad))
    canvas[iy - h + pad:iy + h + 1 + pad, ix - h + pad:ix + h + 1 + pad] = sh
    return canvas[pad:pad + shape[0], pad:pad + shape[1]]


def _moffat_stamp(fwhm, beta=3.0):
    return synth.PSFField((200, 200), fwhm=fwhm, beta=beta).stamp(100.0, 100.0).data


def _direct_with_stamp(star_x, star_y, star_flux, star_det, det_x, det_y, amp2_or_amp, stamps_at,
                       sigma, coherent):
    a, own, nb = [], [], []
    shape = sigma.shape
    for s in range(len(star_x)):
        ix, iy = int(np.rint(star_x[s])), int(np.rint(star_y[s]))
        box = (slice(iy - HALF, iy + HALF + 1), slice(ix - HALF, ix + HALF + 1))
        acc = np.zeros(shape)
        for j in range(len(det_x)):
            if j == star_det[s]:
                continue
            pj = _placed(stamps_at(j), det_x[j], det_y[j], shape)
            acc += amp2_or_amp[j] * pj if coherent else amp2_or_amp[j] * pj ** 2
        val = acc[box] ** 2 if coherent else acc[box]
        own.append(float(np.mean(np.asarray(sigma[box], float) ** 2)))
        nb.append(float(np.mean(val)))
        a.append(N_PIX * (own[-1] + nb[-1]) / star_flux[s] ** 2)
    return np.array(a), np.array(own), np.array(nb)


import synth  # noqa: E402


# History: round 2 placed the empirical stamp itself (Fourier-shifted) at each neighbour; rounds 3-4 a piecewise-linear
# circular profile with a Fourier phase table (its far wing rang, impl_a.3.md round 4).  Round 5 (prereg_a.3_r5.md (ga),
# kept by r5d) replaces both by a Moffat fitted to the empirical stamp, placed by a 16 x 16 table of directly integrated
# phase stamps; its definition is tested in test_psf_iter0_fit.py, its table accuracy in test_prereg_a3_r5d.py (P-A2).

def _noisy_moffat_stamp(fwhm, seed):
    rng = np.random.default_rng(seed)
    st = _moffat_stamp(fwhm)
    st = st + rng.normal(0.0, 2e-5, st.shape)          # a noise floor that the 3-SE cut finds inside the box
    return st / st.sum()


@pytest.mark.dev_record                  # reads the registered truth module through test_psf_iter0_fit
def test_iteration0_with_the_radial_profile_matches_the_formula():
    """psf0_star_noise with a profile (the energy step of prereg_a.3_r5d.md 1.4): each neighbour is the profile placed by
    its phase table (bilinear blend of the four nearest of the 17 x 17 directly integrated phase stamps, centre on
    (floor x, floor y)); sigma2_nb = box mean of the summed light squared.  Expected: the registered reference table
    (prereg_a3_r5d_truth.MoffatTable, sha256 asserted) with the profile's (fwhm, beta), placed pixel by pixel here."""
    import test_psf_iter0_fit as T0
    TR = T0._truth()
    sc = _scene()
    prof = pfs.radial_profile(_noisy_moffat_stamp(4.3, 4))
    table = TR.MoffatTable(prof.fwhm, prof.beta)
    m = table.m

    def neighbour_image(xj, yj, shape):
        st, nx0, ny0 = table.placed(xj, yj)
        pad = 2 * m + 2
        canvas = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad))
        canvas[ny0 - m + pad:ny0 + m + 1 + pad, nx0 - m + pad:nx0 + m + 1 + pad] = st
        return canvas[pad:pad + shape[0], pad:pad + shape[1]]

    want_a, want_own, want_nb = [], [], []
    for s in range(len(sc["star_x"])):
        ix, iy = int(np.rint(sc["star_x"][s])), int(np.rint(sc["star_y"][s]))
        light_img = np.zeros(sc["sigma"].shape)
        for j in range(len(sc["det_x"])):
            if j != sc["star_det"][s]:
                light_img += sc["det_flux"][j] * neighbour_image(sc["det_x"][j], sc["det_y"][j],
                                                                 sc["sigma"].shape)
        light = light_img[iy - HALF:iy + HALF + 1, ix - HALF:ix + HALF + 1]
        own = float(np.mean(np.asarray(sc["sigma"][iy - HALF:iy + HALF + 1, ix - HALF:ix + HALF + 1]) ** 2))
        want_own.append(own)
        want_nb.append(float(np.mean(light ** 2)))
        want_a.append(N_PIX * (own + want_nb[-1]) / sc["star_flux"][s] ** 2)
    got = pfs.psf0_star_noise(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"], sc["det_x"],
                              sc["det_y"], sc["det_flux"], sc["sigma"], FWHM, half=HALF, profile=prof)
    np.testing.assert_allclose(got[1], np.array(want_own), rtol=1e-12)
    # the engine normalises the profile by quadrature, the reference by a fine trapezoid (relative ~6e-10)
    np.testing.assert_allclose(got[2], np.array(want_nb), rtol=1e-8, atol=1e-300)
    np.testing.assert_allclose(got[0], np.array(want_a), rtol=1e-8)
    assert np.max(np.array(want_nb) / (np.array(want_own) + np.array(want_nb))) > 0.5


def test_radial_profile_weight_has_no_jump_as_a_neighbour_moves():
    sigma = np.ones((260, 260))
    prof = pfs.radial_profile(_noisy_moffat_stamp(4.3, 5))
    prev = None
    for dx in np.arange(20.0, 60.0, 0.25):
        w = 1.0 / pfs.psf0_star_noise(np.array([130.2]), np.array([129.7]), np.array([2.0e4]), np.array([0]),
                                      np.array([130.2, 130.2 + dx]), np.array([129.7, 133.0]),
                                      np.array([2.1e4, 1e6]), sigma, FWHM, half=HALF, profile=prof)[0][0]
        if prev is not None:
            assert w >= prev * (1.0 - 1e-12), dx           # moving away never lowers the weight
            assert w <= prev * 1.5, dx                      # and changes it smoothly (0.25 px steps)
        prev = w


def test_auto_flux_aperture_radius():
    """sep_auto_flux(return_radius=True): R_j = the radius of the circle with the Kron ellipse's area
    (2.5 r_Kron sqrt(a b); 1.75 px for the circle fallback).  (The growth correction that used it was removed in
    round 5c/5d: the iteration-0 neighbour amplitudes are fitted, prereg_a.3_r5d.md 1.4.)"""
    import sep
    h, w = 200, 200
    mof = synth.PSFField((h, w), fwhm=4.5)
    truth = mof.stamp(100.0, 100.0)
    rng = np.random.default_rng(8)
    img = np.zeros((h, w))
    img[truth.iy0 - 27:truth.iy0 + 28, truth.ix0 - 27:truth.ix0 + 28] = 1e5 * truth.data
    img += rng.normal(0.0, 1.0, img.shape)
    bad = np.zeros(img.shape, bool)
    objs = pfs.detect_sep_objects(img, np.ones_like(img), bad, thresh_sigma=5.0, minarea=25)
    auto, radius = pfs.sep_auto_flux(img, objs, bad, return_radius=True)
    kron, _ = sep.kron_radius(np.ascontiguousarray(img), objs["x"], objs["y"], objs["a"], objs["b"],
                              objs["theta"], 6.0)
    r_want = np.where(kron * np.sqrt(objs["a"] * objs["b"]) < 1.75, 1.75,
                      2.5 * kron * np.sqrt(objs["a"] * objs["b"]))
    np.testing.assert_allclose(radius, r_want, rtol=1e-12)
    np.testing.assert_array_equal(auto, pfs.sep_auto_flux(img, objs, bad))


def test_empirical_stamp_is_the_median_of_the_brightest_isolated_stamps():
    rng = np.random.default_rng(5)
    st = rng.uniform(0.5, 1.5, (80, 55, 55))
    st /= st.sum(axis=(1, 2), keepdims=True)
    flux = rng.uniform(1.0, 100.0, 80)
    iso = rng.uniform(size=80) < 0.8
    got = pfs.empirical_star_stamp(st, flux, iso)
    top = np.flatnonzero(iso)[np.argsort(-flux[iso])][:50]
    want = np.median(st[top], axis=0)
    want /= want.sum()
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=0.0)
    few = np.zeros(80, dtype=bool)
    few[:4] = True
    assert pfs.empirical_star_stamp(st, flux, few) is None          # < 5 isolated: Gaussian fallback


def test_isolated_means_no_other_detection_peak_in_the_box():
    dt = np.dtype([("x", "f8"), ("y", "f8"), ("xpeak", "i8"), ("ypeak", "i8")])
    objs = np.array([(100.2, 100.4, 100, 100), (120.0, 90.0, 120, 90),      # 0 has 1 in its box
                     (300.0, 300.0, 300, 300), (327.0, 300.0, 328, 300)], dtype=dt)  # 3's peak 28 px out
    got = pfs.isolated_in_box(objs, np.array([0, 2]), half=HALF)
    np.testing.assert_array_equal(got, [False, True])


def test_sep_auto_flux_captures_a_gaussian_and_beats_the_isophotal_flux():
    h, w = 200, 200
    yy, xx = np.mgrid[0:h, 0:w]
    s = 1.8
    img = 1e5 * np.exp(-0.5 * ((xx - 100.3) ** 2 + (yy - 99.6) ** 2) / s ** 2) / (2 * np.pi * s * s)
    rng = np.random.default_rng(2)
    img = img + rng.normal(0.0, 1.0, img.shape)
    sig = np.ones_like(img)
    bad = np.zeros(img.shape, bool)
    objs = pfs.detect_sep_objects(img, sig, bad, thresh_sigma=5.0, minarea=25)
    k = int(np.argmax(objs["flux"]))
    auto = pfs.sep_auto_flux(img, objs, bad)
    # Kron 2.5 of a circular Gaussian: radius 2.5 sqrt(pi/2) sigma, enclosed 1 - exp(-4.909) = 0.9926
    assert 0.985 <= auto[k] / 1e5 <= 0.998, auto[k]
    # a faint Moffat neighbour (total 1e3, sky sigma 1): the isophote cuts its wings, AUTO less so.
    # (Corrected after the first run: for a bright one, 1e5, the 5-sigma isophote reaches far into
    # the wings -- isophotal 0.994 of the total, AUTO 0.954 -- so "AUTO > isophotal" holds only for
    # faint neighbours; impl_a.3.md round 2.)
    mof = synth.PSFField((h, w), fwhm=4.5).stamp(100.0, 100.0)
    img2 = np.zeros((h, w))
    img2[mof.iy0 - 27:mof.iy0 + 28, mof.ix0 - 27:mof.ix0 + 28] = 1e3 * mof.data
    img2 += rng.normal(0.0, 1.0, img2.shape)
    o2 = pfs.detect_sep_objects(img2, sig, bad, thresh_sigma=5.0, minarea=25)
    k2 = int(np.argmax(o2["flux"]))
    a2 = pfs.sep_auto_flux(img2, o2, bad)
    assert a2[k2] > o2["flux"][k2] and a2[k2] / 1e3 >= 0.9, (a2[k2], o2["flux"][k2])


def test_later_iteration_noise_is_the_independent_sum_of_squares():
    sc = _scene()
    rng = np.random.default_rng(9)
    n = len(sc["det_x"])
    sigma_f = rng.uniform(1.0, 50.0, n)
    s_b = rng.uniform(0.0, 0.05, n)
    stamp_at = lambda j: _moffat_stamp(4.0 + 0.01 * sc["det_x"][j])     # PSF varies with position
    c2 = sigma_f ** 2 + (s_b * sc["det_flux"]) ** 2
    want = _direct_with_stamp(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"], sc["det_x"],
                              sc["det_y"], c2, stamp_at, sc["sigma"], coherent=False)
    psf_at = lambda x, y: _moffat_stamp(4.0 + 0.01 * x)
    got = pfs.star_noise_t(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"], sc["det_x"],
                           sc["det_y"], sigma_f, sc["det_flux"], sc["sigma"], psf_at, nb_s_b=s_b, half=HALF)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-300)
    # s_b^(-1) = 0: only the flux errors
    got0 = pfs.star_noise_t(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"], sc["det_x"],
                            sc["det_y"], sigma_f, sc["det_flux"], sc["sigma"], psf_at, half=HALF)
    want0 = _direct_with_stamp(sc["star_x"], sc["star_y"], sc["star_flux"], sc["star_det"], sc["det_x"],
                               sc["det_y"], sigma_f ** 2, stamp_at, sc["sigma"], coherent=False)
    np.testing.assert_allclose(got0[2], want0[2], rtol=1e-12, atol=1e-300)


def test_box_sum_kappa_of_a_correlated_sky():
    """kappa = variance of empty-box sums / (n_pix sigma^2) (3rd ed. 7.1): the synth sky (separable
    3-tap kernel, rho(1 px) = 0.15) has kappa = (sum k)^4 = 1.72 (designer); 3888 empty boxes of
    55 px measure it to a few %."""
    rng = np.random.default_rng(12)
    img = synth.sky_noise((2970, 3960), rng, sigma=1.0)
    no_objs = np.zeros(0, dtype=np.dtype([("x", "f8"), ("y", "f8"), ("xpeak", "i8"), ("ypeak", "i8")]))
    kappa, n_boxes = pfs.box_sum_kappa(img, np.ones_like(img), no_objs, np.zeros(img.shape, bool), half=HALF)
    assert n_boxes == 54 * 72, n_boxes
    assert abs(kappa - 1.72) <= 0.2, kappa
