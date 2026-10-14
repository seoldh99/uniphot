"""Self-tests of the synthetic truth generator (M2 unit 0.3; design_v2.md section 6).

Expected values are computed here from the design's formulas (Moffat, Gaussian, ring, FWHM map,
ring weight, box flux convention, noise kernel) with methods independent of the generator's
(adaptive scipy quadrature, exact erf pixel integrals, lag sums), or directly from the stamps the
generator renders (design 6 E1/A4: "computed directly from the PSF the test uses").  No engine function is used.
The statistical checks live in synth/checks.py so that the false-failure device runs exactly
the same code; their seeds are fixed here before any run.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest
from scipy import integrate, special, stats

import synth
from synth import checks, falsefail

SEED = 20261010
LN2 = math.log(2.0)


# ---------------------------------------------------------------- independent reference formulas
def moffat_ref(u, v, fwhm, beta):
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    return (beta - 1.0) / (math.pi * alpha ** 2) * (1.0 + (u * u + v * v) / alpha ** 2) ** (-beta)


def gauss_ref(u, v, fwhm):
    s = fwhm / (2.0 * math.sqrt(2.0 * LN2))
    return np.exp(-(u * u + v * v) / (2.0 * s * s)) / (2.0 * math.pi * s * s)


def ring_ref(u, v, radius=10.0, sig=1.5):
    norm = 2.0 * math.pi * (sig ** 2 * math.exp(-radius ** 2 / (2 * sig ** 2))
                            + math.sqrt(2.0 * math.pi) * radius * sig * special.ndtr(radius / sig))
    return np.exp(-(np.hypot(u, v) - radius) ** 2 / (2.0 * sig ** 2)) / norm


def pixel_dblquad(f, i, j, dx, dy, half=27):
    """Integral of f(u, v) over box pixel (i, j) for a source at (dx, dy) from the box centre."""
    u0, v0 = i - half - 0.5 - dx, j - half - 0.5 - dy
    val, _ = integrate.dblquad(lambda v, u: f(u, v), u0, u0 + 1.0, v0, v0 + 1.0,
                               epsabs=1e-15, epsrel=1e-12)
    return val


def erf_gauss_stamp(fwhm, dx, dy, size=55):
    s = fwhm / (2.0 * math.sqrt(2.0 * LN2))
    e = np.arange(size + 1) - size // 2 - 0.5
    px = np.diff(special.ndtr((e - dx) / s))
    py = np.diff(special.ndtr((e - dy) / s))
    return np.outer(py, px)


def lag_sum_variance(w, rho1):
    h, wd = w.shape
    tot = 0.0
    for lx in range(-2, 3):
        for ly in range(-2, 3):
            a = w[max(0, ly):h + min(0, ly), max(0, lx):wd + min(0, lx)]
            b = w[max(0, -ly):h + min(0, -ly), max(0, -lx):wd + min(0, -lx)]
            tot += checks.corr_2d(lx, ly, rho1) * float((a * b).sum())
    return tot


# ---------------------------------------------------------------- PSF
@pytest.mark.parametrize("beta", [2.5, 3.0])
@pytest.mark.parametrize("fwhm", [3.7, 7.7])
def test_moffat_alpha_puts_half_maximum_at_half_fwhm(fwhm, beta):
    alpha = np.float64(synth.moffat_alpha(fwhm, beta))
    assert abs((1.0 + (fwhm / 2.0) ** 2 / alpha ** 2) ** (-beta) - 0.5) < 1e-12


def test_box_center_rounds_like_numpy():
    got = synth.box_center(np.array([10.4, 10.5, 11.5, 0.6, -0.4, 27.49]))
    assert list(got) == [10, 10, 12, 1, 0, 27]


def test_pixel_integration_matches_adaptive_quadrature():
    dx, dy = 0.3, -0.2
    for f in (lambda u, v: moffat_ref(u, v, 3.7, 3.0), lambda u, v: moffat_ref(u, v, 7.7, 2.5),
              lambda u, v: ring_ref(u, v)):
        st = synth.pixel_integrate(f, dx, dy)
        for i, j in ((27, 27), (30, 25), (37, 27), (27, 17), (5, 50)):
            ref = pixel_dblquad(f, i, j, dx, dy)
            assert abs(st[j, i] - ref) <= 1e-10 * ref + 1e-14, (i, j, st[j, i], ref)
        box, _ = integrate.dblquad(lambda v, u: f(u, v), -27.5 - dx, 27.5 - dx, -27.5 - dy, 27.5 - dy,
                                   epsabs=1e-13, epsrel=1e-11)
        assert abs(st.sum() - box) <= 1e-9 * box


@pytest.mark.parametrize("fwhm", [2.5, 4.0, 7.7])
def test_pixel_integration_of_gaussian_matches_exact_erf(fwhm):
    dx, dy = 0.21, -0.37
    st = synth.pixel_integrate(lambda u, v: gauss_ref(u, v, fwhm), dx, dy)
    assert np.max(np.abs(st - erf_gauss_stamp(fwhm, dx, dy))) < 1e-13


def test_quadrature_order_is_converged():
    shape = (3001, 4001)
    lo = synth.PSFField(shape, ring_corner=True)            # default order
    hi = synth.PSFField(shape, ring_corner=True, order=2 * lo.order)
    for x, y in ((2000.3, 1500.2), (4000.0, 0.0), (3900.4, 2930.7), (0.2, 3000.0)):
        a, b = lo.stamp(x, y).data, hi.stamp(x, y).data
        assert abs(b.sum() - 1.0) < 1e-12
        assert np.max(np.abs(a - b)) <= 1e-10 * b.max(), (x, y)


def test_stamps_are_box_normalised_nonnegative_and_centred():
    shape = (3000, 4000)
    field = synth.PSFField(shape, ring_corner=True)
    rng = np.random.default_rng(SEED)
    for x, y in zip(rng.uniform(0, 3999, 40), rng.uniform(0, 2999, 40)):
        st = field.stamp(x, y)
        assert st.data.shape == (55, 55)
        assert abs(st.data.sum() - 1.0) < 1e-12
        assert st.data.min() >= 0.0
        assert (st.ix0, st.iy0) == (int(np.rint(x)), int(np.rint(y)))
        assert 0.99 < st.box_fraction <= 1.0


def test_box_fraction_is_the_continuous_psf_inside_the_box():
    shape = (3001, 4001)
    field = synth.PSFField(shape, ring_corner=True)
    for x, y in ((4000.0, 0.0), (3900.4, 2930.7)):
        st = field.stamp(x, y)
        dx, dy = x - st.ix0, y - st.iy0
        box, _ = integrate.dblquad(lambda v, u: field.continuous(x, y, u, v), -27.5 - dx, 27.5 - dx,
                                   -27.5 - dy, 27.5 - dy, epsabs=1e-13, epsrel=1e-11)
        assert 0.99 < st.box_fraction <= 1.0
        assert abs(st.box_fraction - box) < 1e-9


def test_fwhm_map_and_ring_weight_follow_the_design():
    shape = (3001, 4001)   # pixel coordinates 0 and 4000 / 3000 map to xh, yh = -1 and +1
    f = synth.PSFField(shape, ring_corner=True)
    xh = lambda v: (v + 1.0) / 2.0 * 4000.0
    yh = lambda v: (v + 1.0) / 2.0 * 3000.0
    for a, b in ((1, -1), (-1, 1), (0, 0), (1, 1), (-1, -1), (0.5, -0.25)):
        want = 3.7 + (a - b) ** 2
        assert abs(f.fwhm_at(xh(a), yh(b)) - want) < 1e-12
        assert abs(f.fwhm_at(xh(a), yh(b), fwhm_scale=1.01) - 1.01 * want) < 1e-12
        got = f.xyhat(xh(a), yh(b))
        assert abs(got[0] - a) < 1e-12 and abs(got[1] - b) < 1e-12
    for a, b, w in ((1, 1, 0.6), (0.8, 0.9, 0.3), (0.6, 1, 0.0), (0.5, 0.9, 0.0), (0.9, 0.7, 0.15)):
        assert abs(f.ring_weight_at(xh(a), yh(b)) - w) < 1e-12, (a, b)
    plain = synth.PSFField(shape)
    assert plain.ring_weight_at(xh(1), yh(1)) == 0.0
    const = synth.PSFField(shape, fwhm=4.3)
    assert abs(const.fwhm_at(xh(1), yh(-1)) - 4.3) < 1e-12


def test_continuous_psf_is_the_design_mixture_with_unit_integral():
    shape = (3001, 4001)
    f = synth.PSFField(shape, ring_corner=True)
    x, y = 0.95 * 2000.0 + 2000.0, 0.95 * 1500.0 + 1500.0      # xh = yh = 0.95: w = 0.525, FWHM 3.7
    w = 0.6 * (0.95 - 0.6) / 0.4
    for u, v in ((0.0, 0.0), (1.3, -0.4), (9.8, 1.0), (-20.0, 7.0)):
        want = (1 - w) * moffat_ref(u, v, 3.7, 3.0) + w * ring_ref(u, v)
        assert abs(f.continuous(x, y, u, v) - want) <= 1e-13 * want
    for field in (f, synth.PSFField(shape, profile="gaussian", fwhm=4.0)):
        tot, _ = integrate.quad(lambda r: field.continuous(x, y, r, 0.0) * 2 * math.pi * r, 0, np.inf,
                                epsabs=1e-13, epsrel=1e-11, limit=400)
        assert abs(tot - 1.0) < 1e-9


# ---------------------------------------------------------------- ideal fits and their formulas
def test_poisson_factor_is_the_direct_sum_and_reaches_the_gaussian_limit():
    st = synth.PSFField((55, 55), fwhm=4.0).stamp(27.21, 26.63).data
    assert abs(synth.poisson_factor(st) - (st ** 3).sum() / (st ** 2).sum() ** 2) < 1e-13
    wide = synth.PSFField((55, 55), profile="gaussian", fwhm=12.0).stamp(27.0, 27.0).data
    assert abs(synth.poisson_factor(wide) - 4.0 / 3.0) < 2e-3


def test_flux_variance_and_sky_limited_flux_follow_the_formulas():
    st = synth.PSFField((55, 55), fwhm=4.0).stamp(27.3, 27.1).data
    s2, s3 = (st ** 2).sum(), (st ** 3).sum()
    flux = 5000.0
    assert abs(synth.flux_variance(st, flux, sigma=1.0, gain=50.0, rho1=0.0)
               - (1.0 / s2 + flux * s3 / (50.0 * s2 ** 2))) < 1e-9
    w = st / s2
    sky = lag_sum_variance(w, 0.15)
    assert abs(synth.flux_variance(st, flux, sigma=2.0, gain=None, rho1=0.15) - 4.0 * sky) < 1e-9
    assert abs(synth.linear_estimator_variance(w, sigma=1.0, rho1=0.15) - sky) < 1e-12
    f100 = synth.sky_limited_flux(100.0, st, sigma=1.0, rho1=0.15)
    assert abs(f100 / math.sqrt(sky) - 100.0) < 1e-9


def test_fixed_offset_flux_ratio_against_exact_gaussian_stamps():
    shape = (101, 101)
    g = synth.PSFField(shape, profile="gaussian", fwhm=4.3)
    assert abs(synth.fixed_offset_flux_ratio(g, 50.2, 49.9, 0.0, 0.0) - 1.0) < 1e-12
    # star at (50.2, 49.9), model held at (50.7, 49.9): both boxes on a common canvas
    star = np.zeros(shape)                           # rint(50.2) = 50, rint(49.9) = 50
    s = erf_gauss_stamp(4.3, 0.2, -0.1)
    star[50 - 27:50 + 28, 50 - 27:50 + 28] = s / s.sum()
    model = np.zeros(shape)
    m = erf_gauss_stamp(4.3, 0.7 - 1.0, -0.1)       # rint(50.7) = 51
    model[50 - 27:50 + 28, 51 - 27:51 + 28] = m / m.sum()
    want = (model * star).sum() / (model ** 2).sum()
    assert abs(synth.fixed_offset_flux_ratio(g, 50.2, 49.9, 0.5, 0.0) - want) < 1e-12
    mof = synth.PSFField(shape, fwhm=4.3)
    plus = synth.fixed_offset_flux_ratio(mof, 50.0, 50.0, 0.5, 0.0)
    minus = synth.fixed_offset_flux_ratio(mof, 50.0, 50.0, -0.5, 0.0)
    assert abs(plus - minus) < 1e-12 and 0.9 < plus < 1.0


# ---------------------------------------------------------------- noise (deterministic parts)
def test_kernel_and_autocorrelation_follow_the_independent_derivation():
    a = checks.kernel_a(0.15)
    k = synth.corr_kernel_1d(0.15)
    assert np.allclose(k, np.array([a, 1.0, a]) / math.sqrt(1 + 2 * a * a), rtol=0, atol=1e-15)
    for lx in range(-3, 4):
        for ly in range(-3, 4):
            assert abs(synth.autocorrelation(lx, ly, 0.15) - checks.corr_2d(lx, ly, 0.15)) < 1e-15
    assert abs(synth.autocorrelation(1, 0, 0.15) - 0.15) < 1e-15
    assert abs(synth.autocorrelation(0, 1, 0.0)) == 0.0


def test_realize_without_noise_and_poisson_only_is_exact():
    model = synth.render((80, 90), [40.2], [39.7], [1000.0], synth.PSFField((80, 90), fwhm=4.0))[0]
    rng = np.random.default_rng(SEED)
    assert np.array_equal(synth.realize(model, rng, sigma=0.0, gain=None), model)
    pois = synth.realize(model, rng, sigma=0.0, gain=50.0)
    assert np.max(np.abs(pois * 50.0 - np.rint(pois * 50.0))) < 1e-9
    stack = synth.realize(model, rng, n=3)
    assert stack.shape == (3, 80, 90)
    assert not np.array_equal(stack[0], stack[1])


# ---------------------------------------------------------------- scene, seeds, XP
def test_render_conserves_box_flux_and_is_linear():
    shape = (200, 300)
    field = synth.PSFField(shape, ring_corner=True)
    x, y, f = np.array([60.3, 150.7, 240.1]), np.array([50.2, 120.9, 160.4]), np.array([1e3, 5e3, 2e4])
    model, info = synth.render(shape, x, y, f, field)
    assert abs(model.sum() - f.sum()) <= 1e-12 * f.sum()
    parts = sum(synth.render(shape, x[k:k + 1], y[k:k + 1], f[k:k + 1], field)[0] for k in range(3))
    assert np.max(np.abs(model - parts)) <= 1e-12 * model.max()
    for k in range(3):
        st = field.stamp(x[k], y[k])
        assert abs(info["sumsq"][k] - (st.data ** 2).sum()) < 1e-15
        assert abs(info["sumcube"][k] - (st.data ** 3).sum()) < 1e-15
        assert (info["ix0"][k], info["iy0"][k]) == (st.ix0, st.iy0)
    edge, _ = synth.render(shape, [5.0], [100.0], [1000.0], field)
    assert 400.0 < edge.sum() < 1000.0


def test_make_image_is_reproducible_and_matches_render():
    shape = (120, 160)
    field = synth.PSFField(shape)
    x, y, f = np.array([60.3, 100.7]), np.array([50.2, 70.9]), np.array([1e3, 5e3])
    a = synth.make_image(shape, x, y, f, field, np.random.default_rng(1))
    b = synth.make_image(shape, x, y, f, field, np.random.default_rng(1))
    c = synth.make_image(shape, x, y, f, field, np.random.default_rng(2))
    assert np.array_equal(a.data, b.data) and not np.array_equal(a.data, c.data)
    assert np.array_equal(a.model, synth.render(shape, x, y, f, field)[0])
    assert np.array_equal(a.x, x) and np.array_equal(a.flux, f)
    assert (a.sigma, a.gain, a.rho1) == (1.0, 50.0, 0.15)


def test_counts_positions_and_snr_draws():
    assert synth.star_count((1000, 1000), "sparse", n=500) == 500
    for bad in (None, 100, 5000):
        with pytest.raises(ValueError):
            synth.star_count((1000, 1000), "sparse", n=bad)
    rng = np.random.default_rng(SEED)
    x, y = synth.draw_positions((300, 400), 5000, rng)
    assert x.min() >= 27 and x.max() <= 399 - 27 and y.min() >= 27 and y.max() <= 299 - 27
    s = synth.draw_snr(5000, 50.0, 2000.0, rng)
    assert s.min() >= 50.0 and s.max() <= 2000.0
    assert abs(np.median(np.log(s)) - 0.5 * (math.log(50.0) + math.log(2000.0))) < 0.1


def test_offset_a1_values():
    shape = (3001, 4001)
    off = synth.offset_a1(shape)
    for (a, b), (wx, wy) in {(0, 0): (-0.134, -0.026), (1, 1): (-0.064, 0.044),
                             (-1, -1): (-0.204, -0.096), (0.5, -0.5): (-0.10650, -0.05350)}.items():
        dx, dy = off((a + 1) / 2 * 4000.0, (b + 1) / 2 * 3000.0)
        assert abs(dx - wx) < 1e-12 and abs(dy - wy) < 1e-12, (a, b, dx, dy)


def test_gaia_seeds_errors_range_and_wrong_positions():
    rng = np.random.default_rng(SEED)
    x, y = synth.draw_positions((3000, 4000), 1000, rng)
    s = synth.gaia_seeds(x, y, rng, wrong_fraction=0.05)
    assert s.pos_err.min() >= 0.002 and s.pos_err.max() <= 0.03
    assert int(s.wrong.sum()) == 50
    t = synth.gaia_seeds(x, y, rng, pos_err_range=(1e-12, 1e-12), wrong_fraction=0.05, wrong_offset=0.5)
    d = np.hypot(t.x - x, t.y - y)
    assert np.all(np.abs(d[t.wrong] - 0.5) < 1e-9) and np.all(d[~t.wrong] < 1e-9)


def test_zp_surface_and_xp_without_noise_are_exact():
    shape = (3001, 4001)
    zp = synth.zp_z1(shape)
    for (a, b), want in {(0, 0): 25.0, (1, 0): 25.05, (0, 1): 25.02, (-1, -1): 24.97}.items():
        assert abs(zp((a + 1) / 2 * 4000.0, (b + 1) / 2 * 3000.0) - want) < 1e-12
    rng = np.random.default_rng(SEED)
    x, y = synth.draw_positions(shape, 200, rng)
    flux = 10 ** rng.uniform(2, 5, 200)
    out = synth.synthetic_xp(flux, x, y, zp, rng, sigma_mag=0.0, outlier_fraction=0.05)
    r = out.mag - (zp(x, y) - 2.5 * np.log10(flux))
    assert int(out.outlier.sum()) == 10
    assert np.all(np.abs(r[out.outlier] - 0.3) < 1e-12) and np.all(np.abs(r[~out.outlier]) < 1e-12)


# ---------------------------------------------------------------- statistical checks (fixed seeds)
def test_crowded_count_gives_the_design_density_over_the_drawing_area():
    # review B1: the count must come from the same area the centres are drawn in
    rng = np.random.default_rng(SEED)
    for shape in ((384, 384), (500, 500), (1000, 1000), (3000, 4000)):
        for margin in (27, 0, -27):
            area = (shape[1] - 1 - 2 * margin) * (shape[0] - 1 - 2 * margin)
            n = synth.star_count(shape, "crowded", margin=margin)
            assert abs(n - 3.3e-3 * area) <= 0.5, (shape, margin, n / area)
            x, y = synth.crowded_positions(shape, rng, margin=margin)
            assert len(x) == n == len(y)
            assert x.min() >= margin and x.max() <= shape[1] - 1 - margin
            assert y.min() >= margin and y.max() <= shape[0] - 1 - margin
    assert synth.star_count((500, 500), "crowded") == synth.star_count((500, 500), "crowded", margin=27)


@pytest.mark.parametrize("name", sorted(checks.REGISTRY))
def test_statistical_check_passes(name):
    fn, _ = checks.REGISTRY[name]
    res = fn(np.random.default_rng(SEED))
    assert res.stats, name
    assert res.passed, res.message()


# ---------------------------------------------------------------- positive controls (test power)
# Each statistical check must fail when the generator is deliberately wrong in the property the
# check guards (impl_plan 1.0-2: the power of a new module's tests is shown by positive controls
# and the ideal implementation's false-failure rate).  The breakage is injected with monkeypatch
# here; the generator has no switch for it.
def _broken_kernel(rho_true):
    a = checks.kernel_a(rho_true)
    return lambda rho1=0.15: np.array([a, 1.0, a]) / math.sqrt(1 + 2 * a * a)


def _broken_poisson(scale):
    return lambda model, gain, rng: rng.poisson(scale * gain * np.asarray(model)) / (scale * gain)


@pytest.mark.parametrize("name, target, attr, broken", [
    ("synth-sky-noise", "noise", "corr_kernel_1d", _broken_kernel(0.17)),
    ("synth-sky-weighted-sum", "noise", "corr_kernel_1d", _broken_kernel(0.20)),
    ("synth-poisson", "noise", "poisson_counts", _broken_poisson(0.95)),
    ("E1-ideal", "noise", "poisson_counts", _broken_poisson(0.90)),
])
def test_positive_control_broken_noise_fails_the_check(monkeypatch, name, target, attr, broken):
    monkeypatch.setattr(getattr(synth, target), attr, broken)
    fn, _ = checks.REGISTRY[name]
    assert not fn(np.random.default_rng(SEED)).passed, name


def test_positive_control_misreported_seed_errors_fail_the_check(monkeypatch):
    real = synth.astrometry.gaia_seeds

    def under_reported(*a, **k):        # POS_ERR reported 15% smaller than the scatter used
        s = real(*a, **k)
        return synth.Seeds(s.x, s.y, 0.85 * s.pos_err, s.wrong)

    monkeypatch.setattr(synth.astrometry, "gaia_seeds", under_reported)
    assert not checks.check_seeds(np.random.default_rng(SEED)).passed


def test_positive_control_biased_xp_fails_the_check(monkeypatch):
    real = synth.xp.synthetic_xp

    def biased(*a, **k):                 # 4 mmag offset = 18 standard errors at n = 2000
        out = real(*a, **k)
        return synth.XP(out.mag + 0.004, out.outlier)

    monkeypatch.setattr(synth.xp, "synthetic_xp", biased)
    assert not checks.check_xp(np.random.default_rng(SEED)).passed


# ---------------------------------------------------------------- false-failure device
def _toy_check(rng):
    z = float(rng.standard_normal())
    return checks.CheckResult("toy", {"z": (z, -2.0, 2.0)})


def test_rngs_for_follows_seedsequence_spawn():
    got = [g.random() for g in falsefail.rngs_for(7, 3)]
    want = [np.random.default_rng(s).random() for s in np.random.SeedSequence(7).spawn(3)]
    assert got == want


def test_measure_counts_failures_and_estimates_the_tail():
    n = 4000
    rec = falsefail.measure("toy", _toy_check, n, seed0=11)
    recount = sum(not _toy_check(g).passed for g in falsefail.rngs_for(11, n))
    assert rec.n_repeats == n and rec.n_fail == recount
    assert rec.rate == recount / n
    tail = 2 * stats.norm.sf(2.0)                         # 0.0455
    assert abs(rec.normal_tail_rate - tail) < 0.008
    assert rec.rate_upper95 >= rec.rate
    assert abs(falsefail.clopper_pearson_upper(0, 300) - (1 - 0.05 ** (1 / 300))) < 1e-12
    assert abs(falsefail.clopper_pearson_upper(5, 100) - stats.beta.ppf(0.95, 6, 95)) < 1e-12


def test_record_round_trip(tmp_path):
    rec = falsefail.measure("toy", _toy_check, 50, seed0=3, note="round trip")
    path = falsefail.save(rec, tmp_path)
    assert path.is_file()
    assert path.name == "toy.json" and json.loads(path.read_text())["n_repeats"] == 50
    back = falsefail.load("toy", tmp_path)
    assert back == rec


def test_generator_hash_ignores_docstrings_and_comment_lines_but_not_code(tmp_path):
    # review B2: a stale record must fail, but a documentation-only edit must not force a re-measure
    src = synth.psf.__file__
    pkg = tmp_path / "synth"
    pkg.mkdir()
    for p in sorted(pathlib.Path(src).parent.glob("*.py")):
        (pkg / p.name).write_text(p.read_text())
    h0 = falsefail.generator_sha256(pkg)
    f = pkg / "psf.py"
    text = f.read_text()
    f.write_text(text.replace('"""Pixel index of the box centre', '"""(edited) Pixel index of the box centre')
                 .replace("BOX = 55\n", "# a new comment-only line\nBOX = 55\n"))
    assert f.read_text() != text
    assert falsefail.generator_sha256(pkg) == h0
    f.write_text(text.replace("BOX = 55\n", "BOX = 57\n"))
    assert falsefail.generator_sha256(pkg) != h0
    assert falsefail.generator_sha256() == falsefail.generator_sha256(pathlib.Path(src).parent)


def test_measure_records_the_hash_of_the_check_source():
    rec = falsefail.measure("toy", _toy_check, 5, seed0=1)
    assert len(rec.check_sha256) == 64
    assert rec.check_sha256 == falsefail.source_sha256(__file__)


def test_acceptance_is_the_design_one_percent_rule():
    # review B3: both the normal tail and the counted rate's 95% upper bound must be below 1%
    def rec(upper, tail):
        return falsefail.Record("x", 500, 2, 0.004, upper, tail)
    assert falsefail.accepts(rec(0.0099, 0.0099))[0]
    assert not falsefail.accepts(rec(0.0125, 0.0055))[0]        # E1 at 500 repeats
    assert not falsefail.accepts(rec(0.0050, 0.0120))[0]
    assert not falsefail.accepts(rec(0.0100, 0.0010))[0]        # strict
    ok, why = falsefail.accepts(rec(0.0149, 0.0002))
    assert not ok and any("upper" in w for w in why)


@pytest.mark.parametrize("name", sorted(checks.REGISTRY))
def test_false_failure_record_exists_and_is_below_one_percent(name):
    _, repeats = checks.REGISTRY[name]
    rec = falsefail.load(name)
    assert rec.n_repeats >= repeats, (name, rec.n_repeats)
    assert rec.generator_sha256 == falsefail.generator_sha256(), (
        f"record {name} was measured on another generator source; re-measure: cd tract7dt/tests && "
        "python -m synth.falsefail --workers 12 --write (E1-ideal with its recorded --repeats)")
    ok, why = falsefail.accepts(rec)
    assert ok, (name, why)
