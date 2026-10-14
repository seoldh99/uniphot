"""Iteration-0 neighbour term of M2 unit a.3, round 5d: wiring tests (prereg_a.3_r5d.md 4 item 5).

Registration: development record prereg_a.3_r5d.md (not distributed) 1.1-1.3 with its addendum A.3 (projected position
iterations) and the round-5 (ga) Moffat profile (prereg_a.3_r5.md 1).  The registered tests (P-A1..P-A5 and the 1.5
conformance check) are in test_prereg_a3_r5d.py; these are the small checks that follow from the definition:

- the round-5 (ga) profile: pixel-integrated (Gauss-Legendre 3 x 3) Moffat (fwhm, beta), support
  R = alpha sqrt(10^(5/beta) - 1), cosine taper 0.8 R..R, unit plane integral, a 17 x 17 phase table, bilinear blend;
- known sources: list rows within R_DUP_FWHM (0.10) x the Moffat FWHM merged (single linkage), each component at the
  mean of its rows, ordered by the smallest row index; detections with no row within R_U_FWHM (1.0) x the FWHM mode
  (<= counts as listed) are sources at their centroids;
- the box fit: own column + two shift columns (h = 0.25 px), neighbour columns with a ridge (A_u / F_u)^2, a neighbour
  dropped when ||c_u||_w = 0 or F_u ||c_u||_w < T_DROP (0.01), one constant; F_u = 10 (max(0, 3 x 3 max) + 5 sigma) /
  phi_max, +inf when one of the nine pixels has zero weight or is outside the image;
- the own position: projected Gauss-Newton iterations on the 1 px disc around the measured centre (stop when the next
  step is <= 0.01 px, at most 5 updates, own amplitude <= 0 stops).

Expected values come from the definition and from the registered reference implementation
(prereg_a3_r5d_truth.py, asserted by sha256), never from the module under test.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import math

import numpy as np
import pytest

import dev_records
from tract7dt.fit.grid_gpu import psf_from_stars as pfs

TRUTH_NAME = "prereg_a3_r5d_truth.py"
TRUTH_SHA256 = "19bef25e018f8dfc2af3b1f9f97bded1a0dc3c6103789ccf900e36e675725c3a"
HALF = 27


@functools.lru_cache(maxsize=1)
def _truth():
    truth_path = dev_records.plan_file(TRUTH_NAME)
    got = hashlib.sha256(open(truth_path, "rb").read()).hexdigest()
    assert got == TRUTH_SHA256, f"prereg_a3_r5d_truth.py sha256 {got} != registered {TRUTH_SHA256}"
    spec = importlib.util.spec_from_file_location("prereg_a3_r5d_truth_unit", truth_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(table, srcs, shape):
    """Image of point sources (x, y, flux) drawn with a phase-table profile (the reference's `placed`)."""
    h, w = shape
    m = table.m
    img = np.zeros((h + 2 * m + 2, w + 2 * m + 2))
    for x, y, f in srcs:
        st, jx, jy = table.placed(x, y)
        img[jy + 1:jy + 2 * m + 2, jx + 1:jx + 2 * m + 2] += f * st
    return img[m + 1:m + 1 + h, m + 1:m + 1 + w]


def _fit(img, sky, zw, cx, cy, start, own, src_x, src_y, F, prof, **kw):
    return pfs.fit_iteration0_boxes(img, sky, zw, np.array([cx]), np.array([cy]), np.array([start[0]]),
                                    np.array([start[1]]), np.array([own]), np.asarray(src_x, float),
                                    np.asarray(src_y, float), np.asarray(F, float), prof, half=HALF, **kw)


# ---------------------------------------------------------------------------------------------------------------
# The round-5 (ga) profile
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.dev_record
@pytest.mark.parametrize("fwhm,beta", [(4.5, 3.0), (3.369, 2.069), (6.0, 2.5)])
def test_moffat_profile_follows_the_round5_definition(fwhm, beta):
    from scipy import integrate
    TR = _truth()
    ref = TR.MoffatTable(fwhm, beta)
    prof = pfs.MoffatProfile(fwhm, beta)
    alpha = fwhm / (2.0 * math.sqrt(2.0 ** (1.0 / beta) - 1.0))
    assert prof.r_sup == pytest.approx(alpha * math.sqrt(10.0 ** (5.0 / beta) - 1.0), rel=1e-14)
    assert (prof.r_cut, prof.m) == (ref.r_cut, ref.m)
    # unit plane integral (independent quadrature; the taper has kinks at 0.8 R and R)
    val, _ = integrate.quad(lambda r: 2.0 * math.pi * r * float(prof.func(np.array(r), np.array(0.0))), 0.0,
                            prof.r_sup, points=[0.8 * prof.r_sup], limit=400, epsabs=0.0, epsrel=1e-13)
    assert val == pytest.approx(1.0, abs=1e-11)
    # the phase table: the reference's table up to its trapezoid normalisation (relative ~1e-9)
    assert prof.tab.shape == ref.tab.shape == (17, 17, 2 * ref.m + 1, 2 * ref.m + 1)
    np.testing.assert_allclose(prof.tab, ref.tab, rtol=0.0, atol=1e-7 * float(ref.tab.max()))
    assert prof.peak == float(np.max(prof.tab[0, 0]))
    rng = np.random.default_rng(77)
    for x, y in rng.uniform(50.0, 60.0, (20, 2)):
        st, nx0, ny0 = prof.placed(x, y)
        st_r, nx_r, ny_r = ref.placed(x, y)
        assert (nx0, ny0) == (nx_r, ny_r) == (math.floor(x), math.floor(y))
        np.testing.assert_allclose(st, st_r, rtol=0.0, atol=1e-7 * float(ref.tab.max()))


@pytest.mark.dev_record
def test_radial_profile_is_the_moffat_fit_of_the_stamp():
    import synth
    TR = _truth()
    rng = np.random.default_rng(5)
    st = synth.PSFField((200, 200), fwhm=4.3, beta=3.0).stamp(100.0, 100.0).data
    st = st + rng.normal(0.0, 2e-5, st.shape)
    st = st / st.sum()
    fw, be = TR.fit_moffat(st)
    prof = pfs.radial_profile(st)
    assert isinstance(prof, pfs.MoffatProfile)
    assert abs(prof.fwhm / fw - 1.0) <= 1e-4 and abs(prof.beta / be - 1.0) <= 1e-4, ((prof.fwhm, prof.beta), (fw, be))
    assert abs(fw / 4.3 - 1.0) < 0.02 and abs(be / 3.0 - 1.0) < 0.05


# ---------------------------------------------------------------------------------------------------------------
# Known sources (1.1)
# ---------------------------------------------------------------------------------------------------------------
def test_known_sources_merge_duplicate_rows_and_add_unlisted_detections():
    fw, fm = 4.0, 6.5                          # r_u = 6.5 px exactly (the <= boundary is tested)
    r = 0.10 * fw
    rx = np.array([50.0, 50.0 + 0.9 * r, 80.0, 80.0 + 1.1 * r, 120.0, 120.0 + 0.8 * r, 120.0 + 1.6 * r, 200.0, 200.0])
    ry = np.array([50.0, 50.0, 80.0, 80.0, 40.0, 40.0, 40.0, 10.0, 10.0])
    # rows 7 and 8 coincide (a duplicate entry): one source
    dx = np.array([50.3, 150.0, 80.0 - fm, 200.0 + fm + 1e-6])
    dy = np.array([50.1, 150.0, 80.0, 10.0])
    U = pfs.known_sources(rx, ry, dx, dy, fwhm=fw, fwhm_mode=fm)
    np.testing.assert_array_equal(U.comp_of_row, [0, 0, 1, 2, 3, 3, 3, 4, 4])
    assert U.n_comp == 5 and U.n_rows_merged == 7
    np.testing.assert_allclose(U.x[:5], [50.0 + 0.45 * r, 80.0, 80.0 + 1.1 * r, 120.0 + 0.8 * r, 200.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(U.y[:5], [50.0, 80.0, 80.0, 40.0, 10.0], rtol=0, atol=1e-12)
    # detection 0 -> row 0's component; 1 unlisted; 2 at exactly r_u from row 2 -> listed; 3 just beyond r_u -> unlisted
    np.testing.assert_array_equal(U.unlisted, [1, 3])
    np.testing.assert_array_equal(U.det_src, [0, 5, 1, 6])
    np.testing.assert_allclose(U.x[5:], [150.0, 200.0 + fm + 1e-6])
    assert U.r_dup == pytest.approx(r, rel=1e-15)


def test_plausible_amplitude_and_its_infinite_cases():
    rng = np.random.default_rng(3)
    img = rng.normal(0.0, 1.0, (40, 50))
    img[20, 30] = 55.0
    sky = 1.0 + 0.01 * np.arange(50)[None, :] * np.ones((40, 1))
    zw = np.zeros(img.shape, bool)
    zw[5, 5] = True
    x = np.array([30.2, 4.6, 0.4, 49.0, 10.0])
    y = np.array([19.8, 5.4, 20.0, 20.0, 10.0])
    F = pfs.plausible_amplitude(img, sky, zw, x, y, 0.04)
    want0 = 10.0 * (55.0 + 5.0 * float(sky[19:22, 29:32].max())) / 0.04
    assert F[0] == pytest.approx(want0, rel=1e-14)
    assert np.isinf(F[1])                                   # a zero-weight pixel among the nine
    assert np.isinf(F[2]) and np.isinf(F[3])                # nine pixels not all in the image
    blk = img[9:12, 9:12]
    assert F[4] == pytest.approx(10.0 * (max(0.0, float(blk.max())) + 5.0 * float(sky[9:12, 9:12].max())) / 0.04, rel=1e-14)


# ---------------------------------------------------------------------------------------------------------------
# The box fit (1.2) and the own position (addendum A.3)
# ---------------------------------------------------------------------------------------------------------------
def _sanity_case():
    TR = _truth()
    table = TR.MoffatTable(4.5, 3.0)
    prof = pfs.MoffatProfile(4.5, 3.0)
    star = (100.37, 99.62, 5e4)
    nbs = [(104.1, 101.3, 8e3), (90.6, 95.2, 2e3), (130.2, 140.7, 3e4)]   # two inside the box, one far outside
    srcs = [star] + nbs
    data = _render(table, srcs, (200, 200)) + 7.0
    sky = np.ones((200, 200))
    zw = np.zeros((200, 200), bool)
    ux = np.array([s[0] for s in srcs])
    uy = np.array([s[1] for s in srcs])
    F = TR.plausible_amplitude(data, sky, zw, ux, uy, float(np.max(table.tab[0, 0])))
    return TR, table, prof, srcs, data, sky, zw, ux, uy, F


@pytest.mark.dev_record
def test_noiseless_sources_come_back_within_the_ridge_shrink():
    TR, table, prof, srcs, data, sky, zw, ux, uy, F = _sanity_case()
    np.testing.assert_allclose(pfs.plausible_amplitude(data, sky, zw, ux, uy, prof.peak), F, rtol=1e-7)
    o = _fit(data, sky, zw, 100, 100, srcs[0][:2], 0, ux, uy, F, prof, return_fits=True)
    assert abs(o["own_x"][0] - 100.37) <= 1e-6 and abs(o["own_y"][0] - 99.62) <= 1e-6
    assert o["own"][0] == pytest.approx(5e4, rel=1e-7)
    assert o["const"][0] == pytest.approx(7.0, abs=1e-4)
    fit = o["fits"][0]
    amp = dict(zip(fit["src"].tolist(), fit["amp"].tolist()))
    assert set(amp) == {1, 2, 3}
    assert amp[1] == pytest.approx(8e3, rel=2e-6) and amp[2] == pytest.approx(2e3, rel=2e-6)
    # the far neighbour (in-box S/N ~2) is shrunk by the ridge: one-column approximation 1 / (1 + t^2)
    c = TR._cut(*table.placed(ux[3], uy[3]), table.m, 100, 100)
    t = F[3] * math.sqrt(float(np.sum(c * c)))
    shrink = 1.0 - amp[3] / 3e4
    assert shrink == pytest.approx(1.0 / (1.0 + t * t), rel=0.1), (shrink, 1.0 / (1.0 + t * t))
    # sigma2_nb = mean over the box of the fitted neighbour light (own, shift, constant excluded)
    light = sum(a * TR._cut(*table.placed(ux[u], uy[u]), table.m, 100, 100) for u, a in amp.items())
    assert o["sigma2_nb"][0] == pytest.approx(float(np.mean(light * light)), rel=1e-6)


@pytest.mark.dev_record
def test_own_position_is_recovered_from_an_offset_start():
    TR, table, prof, srcs, data, sky, zw, ux, uy, F = _sanity_case()
    for start in ((100.57, 99.42), (100.67, 99.62), (100.37, 99.32)):          # 0.28, 0.3, 0.3 px off
        o = _fit(data, sky, zw, 100, 100, start, 0, ux, uy, F, prof)
        assert math.hypot(o["own_x"][0] - 100.37, o["own_y"][0] - 99.62) <= 0.01, (start, o["own_x"], o["own_y"])
        assert o["converged"][0] and not o["projected"][0] and 1 <= o["n_updates"][0] <= 5
        assert o["own"][0] == pytest.approx(5e4, rel=1e-3)


@pytest.mark.dev_record
def test_own_position_stays_on_the_one_pixel_disc():
    TR, table, prof, srcs, data, sky, zw, ux, uy, F = _sanity_case()
    start = (100.37 + 1.5, 99.62)
    o = _fit(data, sky, zw, 100, 100, start, 0, ux, uy, F, prof)
    d = math.hypot(o["own_x"][0] - start[0], o["own_y"][0] - start[1])
    assert o["projected"][0] and o["at_edge"][0]
    assert d == pytest.approx(1.0, abs=1e-9)
    assert o["n_updates"][0] <= 5


@pytest.mark.dev_record
def test_nonpositive_own_amplitude_stops_the_iterations():
    TR, table, prof, srcs, data, sky, zw, ux, uy, F = _sanity_case()
    dip = 7.0 - _render(table, [(60.3, 140.4, 300.0)], (200, 200))
    o = _fit(dip, sky, zw, 60, 140, (60.3, 140.4), 0, np.array([60.3]), np.array([140.4]), np.array([np.inf]), prof)
    assert o["own"][0] < 0 and o["own_nonpositive"][0]
    assert o["n_updates"][0] == 0 and not o["converged"][0]
    assert (o["own_x"][0], o["own_y"][0]) == (60.3, 140.4)


@pytest.mark.dev_record
def test_neighbour_drop_threshold():
    TR, table, prof, srcs, data, sky, zw, ux, uy, F = _sanity_case()
    c = TR._cut(*table.placed(ux[1], uy[1]), table.m, 100, 100)
    nrm = math.sqrt(float(np.sum(c * c)))
    for factor, kept in ((1.0 - 1e-6, False), (1.0 + 1e-6, True)):
        F2 = F.copy()
        F2[1] = 0.01 / nrm * factor
        o = _fit(data, sky, zw, 100, 100, srcs[0][:2], 0, ux, uy, F2, prof, return_fits=True)
        assert (1 in o["fits"][0]["src"].tolist()) is kept, factor


@pytest.mark.dev_record
def test_zero_weight_pixels_do_not_enter_the_fit():
    TR, table, prof, srcs, data, sky, zw, ux, uy, F = _sanity_case()
    rng = np.random.default_rng(4)
    noisy = data + rng.normal(0.0, 1.0, data.shape)
    base = _fit(noisy, sky, zw, 100, 100, (100.4, 99.6), 0, ux, uy, F, prof)
    bad = noisy.copy()
    zw2 = zw.copy()
    for (yy, xx) in ((97, 103), (110, 90), (120, 125)):         # pixels in the box, none in a 3 x 3 F_u patch
        bad[yy, xx] += 1e6
        zw2[yy, xx] = True
    got = _fit(bad, sky, zw2, 100, 100, (100.4, 99.6), 0, ux, uy, F, prof)
    ref_masked = _fit(noisy, sky, zw2, 100, 100, (100.4, 99.6), 0, ux, uy, F, prof)
    np.testing.assert_allclose(got["sigma2_nb"], ref_masked["sigma2_nb"], rtol=1e-10)
    assert got["sigma2_nb"][0] == pytest.approx(base["sigma2_nb"][0], rel=0.2)


@pytest.mark.dev_record
def test_sigma2_nb_does_not_depend_on_the_neighbour_order():
    TR = _truth()
    table = TR.MoffatTable(3.4, 2.1)
    prof = pfs.MoffatProfile(3.4, 2.1)
    rng = np.random.default_rng(12)
    n = 40
    x = rng.uniform(40.0, 160.0, n)
    y = rng.uniform(40.0, 160.0, n)
    x[0], y[0] = 100.2, 99.7
    f = 10.0 ** rng.uniform(2.5, 4.5, n)
    data = _render(table, list(zip(x, y, f)), (200, 200)) + rng.normal(0.0, 1.0, (200, 200)) + 3.0
    sky = np.ones((200, 200))
    zw = np.zeros((200, 200), bool)
    F = pfs.plausible_amplitude(data, sky, zw, x, y, prof.peak)
    a = _fit(data, sky, zw, 100, 100, (100.25, 99.65), 0, x, y, F, prof)
    perm = rng.permutation(n)
    inv = np.argsort(perm)
    b = _fit(data, sky, zw, 100, 100, (100.25, 99.65), int(inv[0]), x[perm], y[perm], F[perm], prof)
    assert a["n_kept"][0] >= 10
    assert b["sigma2_nb"][0] == pytest.approx(a["sigma2_nb"][0], rel=1e-10)
    assert b["own"][0] == pytest.approx(a["own"][0], rel=1e-10)


def test_neighbour_bins_count_stars_by_the_distance_of_other_sources():
    fw = 4.0
    sx = np.array([10.0, 50.0, 90.0, 130.0])
    sy = np.zeros(4)
    own = np.array([0, 1, 2, 3])
    # sources 0-3 = the stars' own; 4 at 1.0 px (0.25 FWHM) from star 0; 5 at 3 px (0.75) from star 1;
    # 6 at 6 px (1.5) from star 2 and 7 at 9 px (2.25, outside every bin) from star 3
    ux = np.array([10.0, 50.0, 90.0, 130.0, 11.0, 53.0, 96.0, 139.0])
    uy = np.zeros(8)
    inv_a = np.array([1.0, 2.0, 3.0, 4.0])
    b = pfs.iteration0_neighbour_bins(sx, sy, own, ux, uy, fw, inv_a)
    assert [d["n_stars"] for d in b] == [1, 1, 1]
    np.testing.assert_allclose([d["inv_a_share"] for d in b], [0.1, 0.2, 0.3], rtol=1e-15)
    assert [(d["lo"], d["hi"]) for d in b] == [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0)]
